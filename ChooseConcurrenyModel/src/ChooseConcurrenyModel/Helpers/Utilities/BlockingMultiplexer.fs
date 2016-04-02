namespace  Utilities

open System
open System.Collections.Generic
open System.Linq
open System.Collections.Concurrent
open System.Threading

type internal Mode =
    | Starting = 0
    | Running = 1
    | Finished = 2
    | CleanupStarted = 3

// Internal state for each producer
[<Struct>]
type internal ProducerInfo<'T> =

    // producer id 0, 1, 2, ...
    val mutable Index : int 
    // producer's queue
    val mutable Collection : BlockingCollection<'T>
    // true if producer's IsCompleted property was observed to be true 
    val mutable IsCompleted : bool
    // does lookahead value exist?
    val mutable HasPendingValue : bool
    // if yes, lookahead valueValue 
    val mutable PendingValue : 'T
    // if yes, lookahead lock id 
    val mutable PendingLockId : int
    // last lock id read (for error checking only) 
    val mutable LastLockId : int

    new( index:int, collection:BlockingCollection<'T>, isCompleted:bool, 
         hasPendingValue:bool, pendingValue:'T, pendingLockId:int, lastLockId:int) =
         { Index = index; Collection = collection; IsCompleted = isCompleted;
           HasPendingValue = hasPendingValue; PendingValue = pendingValue;
           PendingLockId = pendingLockId; LastLockId = lastLockId }


/// <summary>
///   Multiplexer that serializes inputs from multiple producers into a single 
///   consumer enumeration in a user-specified order. 
/// </summary>
/// <typeparam name="T">The type of input element</typeparam>
/// <remarks>
///   The use case for this class is a producer/consumer scenario with multiple producers and 
///   a single consumer. The producers each have their private blocking collections for enqueuing the elements
///   that they produce. The consumer of the producer queues is the multiplexer, which is responsible 
///   combining the inputs from all of the producers according to user-provided "lock order." The multiplexer 
///   provides an enumeration that a consumer can use to observe the multiplexed values in the chosen order. 
/// 
///   The multiplexer does not perform sorting. Instead, it relies on the fact the the producer queues are
///   locally ordered and looks for the next value by simultaneously monitoring the heads of 
///   all of the producer queues.
/// 
///   The order of elements in the producer queues is given by a user-provided lockOrderFn delegate. This is called
///   lock order and is represented by an integer. The initial lock id is specified in the multiplexer's constructor. 
///   Producer queues must be consistent. This means that they are locally ordered with respect to lock ids. When
///   multiplexed together into a single enumeration, the producer queues must produce a sequence of values whose 
///   lock ids are consecutive. (The lock ids in the individual producer queues must be in order but not necessarily 
///   consecutive.)
/// 
///   It is not required that all elements in the producer queues have a lock order. The position of such elements (denoted
///   by a lock id that is less than zero) is constrained by preceding and succeeding elements in the producer's queue
///   that do include a lock order. This results in a partial order. The unit tests for this class for an example of 
///   partial ordering constraints.
/// 
///   See Campbell et al, "Multiplexing of Partially Ordered Events," in TestCom 2005, Springer Verlag, June 2005,  
///   available online at http://research.microsoft.com/apps/pubs/default.aspx?id=77808. 
/// </remarks>
type BlockingMultiplexer<'T> private (lockOrderFn, initialLockId, boundedCapacity, unitArg) =

    // Private state and constructor initializataion
        
    let producersLock = new obj()
    let mutable nextLockId = initialLockId
    let mutable mode = Mode.Starting
    let mutable producers : ProducerInfo<'T>[] = [| |]

    do
        if initialLockId < 0 then invalidArg "initialLockId" "less than zero"
        if boundedCapacity < -1 then invalidArg "boundedCapacity" "out of range"

    // --------------------------------------------------------------------------
    // Constructors
    // --------------------------------------------------------------------------

    /// <summary>
    /// Creates a multiplexer that serializes inputs from multiple producer queues.
    /// </summary>
    /// <param name="lockOrderFn">Function that returns an integer sequence number for elements of the 
    /// producer queues. It returns a negative number if order is not important for a given element.</param>
    /// <param name="initialLockId">The first lock id of the sequence</param>
    /// <param name="boundedCapacity">The maximum number of elements that a producer queue
    /// may contain before it blocks the producer.</param>
    new(lockOrderFn, initialLockId, boundedCapacity) =
        new BlockingMultiplexer<_>(lockOrderFn, initialLockId, boundedCapacity, ())


    /// <summary>
    /// Creates a multiplexer that serializes inputs from multiple producer queues.
    /// </summary>
    /// <param name="lockOrderFn">Delegate that returns an integer sequence number for elements of the 
    /// producer queues. It returns a negative number if order is not important for a given element.</param>
    /// <param name="initialLockId">The first lock id of the sequence</param>
    new(lockOrderFn, initialLockId) = 
        new BlockingMultiplexer<_>(lockOrderFn, initialLockId, -1)

    // --------------------------------------------------------------------------
    // Public methods
    // --------------------------------------------------------------------------

    /// <summary>
    /// Creates a new input source to the multiplexer.
    /// </summary>
    /// <returns>A blocking collection that will be used as one of the multiplexer's inputs.
    /// </returns>
    /// <remarks>This blocking collection for the use of the producer only. Its only consumer of the 
    /// is the multiplexer instance that created it.
    /// 
    /// The producer should invoke Add to insert elements as needed. After the last element, the producer 
    /// invokes CompleteAdding.
    /// 
    /// If the boundedCapacity was specified in the multiplexer's constructor, this value will be used as the
    /// boundedCapacity of the blocking collections used by the producers. This will cause the producer to block
    /// when trying to add elements to the blocking collection above this limit.
    /// 
    /// There is a partial order constraint on the values added by the producer to this blocking collection. The 
    /// lockOrderFn that was provided to the constructor of the multiplexer will be applied to each element in 
    /// the queue by the multiplexer. If the lockOrderFn returns a non-negative value for the enqueued 
    /// object, this value must be strictly greater than the lock order of all previous objects that were added 
    /// to this blocking collection.
    /// 
    /// All producer queues must be created before getting the consumer's enumerable object.</remarks>
    member x.GetProducerQueue() =
        let result =
            if boundedCapacity <= 0 then new BlockingCollection<_>()
            else new BlockingCollection<_>(boundedCapacity)
        lock producersLock (fun () ->
            if mode <> Mode.Starting then
                raise (new InvalidOperationException("Cannot get new producer queue for running multiplexer"))

            let index = producers.Length
            Array.Resize(&producers, index + 1)
            producers.[index].Index <- index
            producers.[index].Collection <- result
            producers.[index].IsCompleted <- false
            producers.[index].HasPendingValue <- false
            producers.[index].PendingValue <- Unchecked.defaultof<_>
            producers.[index].PendingLockId <- -1
            producers.[index].LastLockId <- -1 )
        result


    /// <summary>
    /// Creates an enumerable object for use by the consumer.
    /// </summary>
    /// <param name="token">The cancellation token</param>
    /// <returns>An enumeration of values. The order of the values will respect the lock order of the
    /// producer queues. This method may be called only one time for this object.</returns>
    member x.GetConsumingEnumerable(?token:CancellationToken) = seq {
        let token = defaultArg token CancellationToken.None
        lock producersLock (fun () ->
            if producers.Length = 0 then
                raise (new InvalidOperationException("Multiplexer requires at least one producer before getting consuming enumerable"))
            if mode <> Mode.Starting then
                raise (new InvalidOperationException("Cannot get enumerator of multiplexer that has already been started"))
            mode <- Mode.Running )

        let complete = ref false
        let yieldBreak = ref false
        while not !yieldBreak && (not !complete || producers.Any(fun info -> info.HasPendingValue)) do

            // Yield case 1: Value with the next lock id is in a lookahead buffer
            if producers.Any(fun info -> info.HasPendingValue && info.PendingLockId = nextLockId) then
                let index = 
                    producers.Single(fun info -> 
                        info.HasPendingValue && info.PendingLockId = nextLockId).Index
                let item = producers.[index].PendingValue

                // clear lookahead buffer
                producers.[index].HasPendingValue <- false
                producers.[index].PendingValue <- Unchecked.defaultof<_>
                producers.[index].PendingLockId <- -1
                producers.[index].LastLockId <- nextLockId

                // consume value
                nextLockId <- nextLockId + 1
                yield item

            // Look ahead values exist but we didn't find the next lock id and there are no 
            // more values to read from the producer queues. This means that producer 
            // blocking collections violated the contract by failing to give all lock ids 
            // between the lowest to the highest observed
            elif !complete then

                // Error occurs only for normal termination, not cancellation
                if not token.IsCancellationRequested then
                    let msg = 
                        "Producer blocking collections completed before giving required lock id "
                        + (nextLockId.ToString())
                        + ". All values up to "
                        + (seq { for info in producers do
                                     if info.HasPendingValue then
                                         yield info.PendingLockId } |> Seq.max |> string)
                        + " are required."
                    raise (new InvalidOperationException(msg))

            else
                let breakLoop = ref false
                while not !complete && not !breakLoop && not !yieldBreak do
                    // Select producers without lookahead values.
                    let waitList = 
                        [| for info in producers do
                               if not info.HasPendingValue && not info.IsCompleted then
                                   yield info.Collection |]

                    if waitList.Length = 0 then
                        if token.IsCancellationRequested then 
                            yieldBreak := true
                        else
                            let msg = "Producer blocking collections omitted required value " + nextLockId.ToString()
                            raise (new InvalidOperationException(msg))
  
                    let item = ref Unchecked.defaultof<_>
                    let waitListIndex = 
                        try BlockingCollection<_>.TakeFromAny(waitList, item)
                        with :? ArgumentException ->
                            // handle occurrence of AddingComplete on another thread.
                            2

                    if waitListIndex < 0 then
                        for i in 0 .. producers.Length - 1 do
                            if not producers.[i].IsCompleted && producers.[i].Collection.IsCompleted then
                                producers.[i].IsCompleted <- true
                        complete := producers.All(fun info -> info.IsCompleted)
                    else
                        let index = 
                           (producers |> Seq.filter (fun info -> info.Collection = waitList.[waitListIndex])
                                      |> Seq.map (fun info -> info.Index)).Single()
                        let lockId = lockOrderFn (!item)

                        // Yield case 2: Item with no ordering constraint. Consume it immediately.
                        if lockId < 0 then yield !item

                        // Yield case 3: Item read is the one we are looking for. Consume it immediately.
                        elif lockId = nextLockId then
                            producers.[index].LastLockId <- lockId
                            nextLockId <- nextLockId + 1
                            yield !item
                            breakLoop := true
                        elif lockId < nextLockId then
                            let msg = "Blocking queue delivered duplicate sequence number to multiplexer (1). The duplicate value is " + lockId.ToString()
                            raise (new InvalidOperationException(msg))
                        elif lockId <= producers.[index].LastLockId then
                            let msg = "Blocking queue delivered out-of-order item (2)"
                            raise (new InvalidOperationException(msg))
                        elif producers |> Seq.filter (fun info -> info.HasPendingValue)
                                       |> Seq.exists (fun info -> info.PendingLockId = lockId) then
                            let msg = "Blocking queue delivered duplicate sequence number to multiplexer (2)"
                            raise (new InvalidOperationException(msg))
                        else
#if DEBUG
                            if producers.[index].HasPendingValue then
                                let msg = "Internal error-- double read from blocking collection"
                                raise (new InvalidOperationException(msg))
#endif
                            producers.[index].HasPendingValue <- true
                            producers.[index].PendingValue <- !item
                            producers.[index].PendingLockId <- lockId

        if not !yieldBreak then
            lock producersLock (fun () ->
                if mode = Mode.Running then
                    mode <- Mode.Finished ) }


    /// Returns an enumeration of all values that have been read by the multiplexer but not yet consumed.
    member x.GetCleanupEnumerable() =
        lock producersLock (fun () ->
            if mode = Mode.Finished || mode = Mode.Running then
                mode <- Mode.CleanupStarted
                [ for  p in producers do
                      if p.HasPendingValue then
                          yield p.PendingValue ]
            else [] )

        
    /// Returns the number of items in all of the producer queues and in the multiplexer's buffers
    member x.Count =
        producers 
            |> Seq.map (fun p ->
                p.Collection.Count + (if p.HasPendingValue then 1 else 0)) 
            |> Seq.sum
