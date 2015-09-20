namespace PerfUtil.CaseStudy

    open FsPickler
    open PerfUtil

    open System.IO
    open System.Runtime.Serialization
    open System.Runtime.Serialization.Formatters.Binary

    [<AbstractClass>]
    type Serializer() =
        let mutable m : MemoryStream = null

        abstract Name : string
        abstract Serialize<'T> : Stream -> 'T -> unit
        abstract Deserialize<'T> : Stream -> 'T

        member s.TestRoundTrip (t : 'T) =
            m.Position <- 0L
            s.Serialize m t
            m.Position <- 0L
            s.Deserialize<'T> m

        interface ITestable with
            member __.Name = __.Name
            member __.Init () = m <- new MemoryStream()
            member __.Fini () = m <- null

    type BFSerializer () =
        inherit Serializer()

        let bfs = new BinaryFormatter()

        override __.Name = "BinaryFormatter"
        override __.Serialize (stream : Stream) (t : 'T) = bfs.Serialize(stream, t)
        override __.Deserialize (stream : Stream) = bfs.Deserialize(stream) :?> 'T

    type NDCSerializer () =
        inherit Serializer()

        let ndc = new NetDataContractSerializer()

        override __.Name = "NetDataContractSerializer"
        override __.Serialize (stream : Stream) (t : 'T) = ndc.Serialize(stream, t)
        override __.Deserialize (stream : Stream) = ndc.Deserialize(stream) :?> 'T

    type FSPSerializer () =
        inherit Serializer ()

        let fsp = new FsPickler()

        override __.Name = "FsPickler"
        override __.Serialize (stream : Stream) (t : 'T) = fsp.Serialize(stream, t)
        override __.Deserialize (stream : Stream) = fsp.Deserialize<'T>(stream)


    type SerializationPerf =

        static member CreateImplementationComparer (?throwOnError, ?warmup) =
            let this = new FSPSerializer() :> Serializer
            let others = [ new BFSerializer() :> Serializer ; new NDCSerializer() :> _ ]
            let comparer = new WeightedComparer(spaceFactor = 0.2, leastAcceptableImprovementFactor = 1.)
            new ImplementationComparer<Serializer>(this, others, comparer = comparer, ?warmup = warmup, ?throwOnError = throwOnError)

        static member CreatePastVersionComparer (historyFile, ?throwOnError, ?warmup) =
            let this = new FSPSerializer () :> Serializer
            let version = typeof<FsPickler.FsPickler>.Assembly.GetName().Version
            let comparer = new WeightedComparer(spaceFactor = 0.2, leastAcceptableImprovementFactor = 0.7)
            new PastImplementationComparer<Serializer>(
                    this, version, historyFile = historyFile, comparer = comparer, ?warmup = warmup, ?throwOnError = throwOnError)
