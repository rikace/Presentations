namespace FSharpWpfMvvmTemplate.ViewModel

open System
open System.Collections.ObjectModel
open System.Windows
open System.Windows.Data
open System.Windows.Input
open SantaClausProblem
open System.ComponentModel
open System.Threading

type UpdateUIMessage =
    | ElfNeedsHelp of string * System.Windows.Threading.Dispatcher
    | SantaHelpsElfes of string * System.Windows.Threading.Dispatcher
    | SantaHelpedElfes of string * System.Windows.Threading.Dispatcher
    | ChristmasYear of int * System.Windows.Threading.Dispatcher
    | ReindeerBack of string * System.Windows.Threading.Dispatcher
    | SantaPreparedSlaigh of string * System.Windows.Threading.Dispatcher
    | SantaDeliveredCompleted of string * System.Windows.Threading.Dispatcher
    | EndOfChristmas of string * System.Windows.Threading.Dispatcher

type MainWindowViewModel() as this =
    inherit ViewModelBase()

    let mutable reindeers = new ObservableCollection<int>()
    let mutable elfes:ObservableCollection<int> = new ObservableCollection<int>()
    let mutable santaSleepingVisibility = Visibility.Visible
    let mutable santaAwakeVisibility = Visibility.Collapsed
    let mutable sleighVisibility = Visibility.Collapsed
    let mutable threeElfesVisibility = Visibility.Collapsed
    let mutable doorOpenVisibility = Visibility.Collapsed
    let mutable doorClosedVisibility = Visibility.Visible
    let mutable title = "Santa Claus Problem"
    let mutable dispatcher = Unchecked.defaultof<Threading.Dispatcher>
    // The cancellation token is used to stop the execution as a whole
    // when the last Year of Christams is reached
    let cancellationTokenSource = new CancellationTokenSource()
    let cancellationToken = cancellationTokenSource.Token

    // Thread-Safe implemantion of random number generator
    // using a MailboxProcessor
    let sleepFor =
        let randomGeneratorAgent = Agent.Start((fun inbox -> async {
            let random = Random(Guid.NewGuid().GetHashCode())
            while true do 
                let! (n, reply:AsyncReplyChannel<int>) = inbox.Receive()
                reply.Reply (random.Next(n)) }), cancellationToken)
        (fun time -> async {
            let! sleepTime = randomGeneratorAgent.PostAndAsyncReply(fun ch -> (time, ch))
            do! Async.Sleep sleepTime })
    
    let updateUsingAppDispatcher (d:System.Windows.Threading.Dispatcher) f = d.Invoke(Action(f))
    // Thread-Safe Logging using a MailboxProcessor
    let updateUI = 
        let updateUIAgent = Agent.Start(fun inbox -> async {
            while true do 
                let! msg = inbox.Receive()
                match msg with
                | ElfNeedsHelp(s, d) -> updateUsingAppDispatcher d (fun () -> (this.Elfes:ObservableCollection<int>).Add(1))
                | SantaHelpsElfes(s, d) -> 
                                        updateUsingAppDispatcher d (fun () -> (this.Elfes:ObservableCollection<int>).Remove(1) |> ignore)
                                        updateUsingAppDispatcher d (fun () -> (this.Elfes:ObservableCollection<int>).Remove(1) |> ignore)
                                        updateUsingAppDispatcher d (fun () -> (this.Elfes:ObservableCollection<int>).Remove(1) |> ignore)
                                        updateUsingAppDispatcher d (fun () -> this.ThreeElfesVisibility <- Visibility.Visible)
                                        updateUsingAppDispatcher d (fun () -> this.DoorClosedVisibility <- Visibility.Collapsed)
                                        updateUsingAppDispatcher d (fun () -> this.DoorOpenVisibility <- Visibility.Visible)
                                        updateUsingAppDispatcher d (fun () -> this.SantaAwakeVisibility <- Visibility.Visible)
                                        updateUsingAppDispatcher d (fun () -> this.SantaSleepingVisibility <- Visibility.Collapsed)
                | SantaHelpedElfes(s, d) -> updateUsingAppDispatcher d (fun () -> this.ThreeElfesVisibility <- Visibility.Collapsed)
                                            updateUsingAppDispatcher d (fun () -> this.DoorClosedVisibility <- Visibility.Visible)
                                            updateUsingAppDispatcher d (fun () -> this.DoorOpenVisibility <- Visibility.Collapsed)
                                            updateUsingAppDispatcher d (fun () -> this.SantaAwakeVisibility <- Visibility.Collapsed)
                                            updateUsingAppDispatcher d (fun () -> this.SantaSleepingVisibility <- Visibility.Visible)
                | ChristmasYear(year, d) -> let title =  sprintf "Santa Claus Problem year %d" year
                                            updateUsingAppDispatcher d (fun () -> this.Title <- title)
                | ReindeerBack(s, d) -> updateUsingAppDispatcher d (fun () -> (this.Reindeers:ObservableCollection<int>).Add(1))  
                | SantaPreparedSlaigh(s, d) -> updateUsingAppDispatcher d (fun () -> this.SleighVisibility <- Visibility.Visible)
                                               updateUsingAppDispatcher d (fun () -> this.SantaAwakeVisibility <- Visibility.Visible)
                                               updateUsingAppDispatcher d (fun () -> this.SantaSleepingVisibility <- Visibility.Collapsed)
                | SantaDeliveredCompleted(s, d) -> updateUsingAppDispatcher d (fun () -> this.SleighVisibility <- Visibility.Collapsed)
                                                   updateUsingAppDispatcher d (fun () -> this.SantaAwakeVisibility <- Visibility.Collapsed)
                                                   updateUsingAppDispatcher d (fun () -> this.SantaSleepingVisibility <- Visibility.Visible)
                                                   updateUsingAppDispatcher d (fun () -> (this.Reindeers:ObservableCollection<int>).Clear())
                | EndOfChristmas(s,d) -> updateUsingAppDispatcher d (fun () -> this.Title <- s) 
                do! Async.Sleep 350 (* slow down the process for animation purpose *)  })
        fun msg -> updateUIAgent.Post msg

    let elvesCount = 10
    let reindeersCount = 9
    let startingYear = ref 2006 
    let theEndOfChristams = 2015
    let elvesNeededToWakeSanta = 3

    // only a given number of elves can wake Santa
    let queueElves = new SyncGate(elvesCount, cancellationToken)
    // SyncGate is used for Santa to prevent a second group of elves 
    // from waking him if the reindeers are waiting and Santa is helping
    // the first group of elves 
    let santasAttention = new SyncGate(1, cancellationToken)
    let allReindeers = new BarrierAsync(reindeersCount, cancellationToken)
    let threeElves = new BarrierAsync(elvesNeededToWakeSanta, cancellationToken)
    let elvesAreInspired = new BarrierAsync(elvesNeededToWakeSanta, cancellationToken)

    let reindeer (id:int) = async { 
        while true do
            do! sleepFor 200
            do! Async.Sleep (reindeersCount * 100)
            // Wait the last Reindeer to be signaled, only all reindeers together
            // can wake Santa Claus to deliver the presents
            let! index = allReindeers.AsyncSignalAndWait() 
            updateUI (ReindeerBack ((sprintf "Reindeer %d is back from vacation... ready to work!" id), Application.Current.Dispatcher))

            // the last reindeer that arrives at the North Pole
            // wakes Santa Claus
            if index = reindeersCount - 1 then  
                // Santa is awake and he is preparing the sleigh,
                // therefore he is busy and cannot help the elves 
                do! santasAttention.AquireAsync()
                updateUI (SantaPreparedSlaigh ((sprintf "Santa has prepared the slaigh"), Application.Current.Dispatcher))
                do! sleepFor 350
                if Interlocked.Increment(startingYear) = theEndOfChristams then
                    cancellationTokenSource.Cancel()
                santasAttention.Release()
                updateUI (SantaDeliveredCompleted ((sprintf "Santa is un-harnessing the reindeer\nAll Reindeers are going back in vacation!"), Application.Current.Dispatcher))
                updateUI (ChristmasYear(!startingYear, Application.Current.Dispatcher))
            do! sleepFor 200 }        
                 
    let elf (id:int) = async {  
        do! sleepFor 2000
        while true do
            // Only 3 elves can open the door of Santa's office
            do! queueElves.AquireAsync() 
            updateUI (ElfNeedsHelp ((sprintf "Elf %d ran out of ideas and he needs to consult santa" id), Application.Current.Dispatcher))
            // Santa waits untill three elves have a problem
            let! index = threeElves.AsyncSignalAndWait()
            // the third elf wakes Santa
            if index = elvesNeededToWakeSanta - 1 then
                do! santasAttention.AquireAsync()
                updateUI (SantaHelpsElfes ("Ho-ho-ho ... some elves are here!\nSanta is consulting with elves...", Application.Current.Dispatcher))
            // wait until all three elves have the solution
            // and inspiration for the toys
            do! sleepFor 500
            let! _ = elvesAreInspired.AsyncSignalAndWait()
            if index = elvesNeededToWakeSanta - 1 then
                santasAttention.Release()
                updateUI (SantaHelpedElfes ("Santa has helped the little elves!\nOK, all done - thanks!", Application.Current.Dispatcher))
            // blocking other elves to disturb Santa.
            queueElves.Release()
            do! sleepFor 2000 }

    let santaClausProblem() =
       
        let reindeers = Array.init reindeersCount (fun i -> reindeer(i))
        let elves  = Array.init elvesCount (fun i -> elf(i))

        cancellationToken.Register(fun () -> 
                (queueElves :> IDisposable).Dispose()
                (allReindeers :> IDisposable).Dispose()
                (threeElves :> IDisposable).Dispose()
                (elvesAreInspired :> IDisposable).Dispose()
                (santasAttention :> IDisposable).Dispose()
                updateUI (EndOfChristmas("Faith has vanished from the world - The End of Santa Claus!!", Application.Current.Dispatcher))) |> ignore

        this.Title <- sprintf "Santa Claus Problem year %d" !startingYear
        let santaJobs =(Seq.append elves reindeers) |> Async.Parallel |> Async.Ignore
        Async.StartImmediate(santaJobs, cancellationToken)
        cancellationTokenSource
                                            
    member x.Start() =        
        santaClausProblem() 
     
    member x.Reindeers 
        with get() = reindeers
        and set value = 
            reindeers <- value 
            x.OnPropertyChanged "Reindeers"

    member x.Elfes 
        with get() = elfes
        and set value = 
            elfes <- value 
            x.OnPropertyChanged "Elfes"

    member x.SantaSleepingVisibility 
        with get() = santaSleepingVisibility
        and set value =
            santaSleepingVisibility <- value
            x.OnPropertyChanged "SantaSleepingVisibility"
            
    member x.SantaAwakeVisibility 
        with get() = santaAwakeVisibility
        and set value =
            santaAwakeVisibility <- value
            x.OnPropertyChanged "SantaAwakeVisibility"

    member x.SleighVisibility 
        with get() = sleighVisibility
        and set value =
            sleighVisibility <- value
            x.OnPropertyChanged "SleighVisibility"

    member x.ThreeElfesVisibility 
        with get() = threeElfesVisibility
        and set value =
            threeElfesVisibility <- value
            x.OnPropertyChanged "ThreeElfesVisibility"

    member x.DoorClosedVisibility 
        with get() = doorClosedVisibility
        and set value =
            doorClosedVisibility <- value
            x.OnPropertyChanged "DoorClosedVisibility"        

    member x.DoorOpenVisibility 
        with get() = doorOpenVisibility
        and set value =
            doorOpenVisibility <- value
            x.OnPropertyChanged "DoorOpenVisibility"  

    member x.Title 
        with get() = title
        and set value =
            title <- value
            x.OnPropertyChanged "Title"