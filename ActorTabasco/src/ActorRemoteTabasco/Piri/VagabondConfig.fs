[<AutoOpen>]
module Piri.VagabondConfig

open System
open System.IO
open Akka.Actor
open Akka.Util
open Nessos.FsPickler
open Nessos.FsPickler.Combinators
open Nessos.Vagabond
open Nessos.Vagabond.ExportableAssembly
open Nessos.Vagabond.AssemblyProtocols

/// Record representing HOCON configuration for Piri plugin
type ServerConfig = 
    { KnownAssemblies: string list
      AssemblyCachePath: string }
    static member Create(config: Akka.Configuration.Config) =
        let asmPath = config.GetString "assembly-cache-path"
        let assemblies = config.GetStringList "known-assemblies"
        { KnownAssemblies = List.ofSeq assemblies; AssemblyCachePath = asmPath }
    static member Create(system: ActorSystem) =
        let config = system.Settings.Config.GetConfig "akka.tabasco"
        ServerConfig.Create config
        
/// Vagabond assembly receiver implementation, which uses ActorRef to the server 
/// to transport assemblies and dependencies between client and server
[<Sealed>]
type internal AssemblyReceiver (vm: VagabondManager, serverRef: IActorRef) =
    member __.ServerRef = serverRef
    interface IRemoteAssemblyReceiver with
        member __.GetLoadedAssemblyInfo (ids:AssemblyId[]) = async {
            let! reply = serverRef <? GetAssemblyInfo ids
            return downcast reply
        }
        member __.PushAssemblies vas = async {
            let! reply = serverRef <? LoadAssemblies (vm.CreateRawAssemblies vas)
            return downcast reply
        } 
    interface IComparable with
        member __.CompareTo(o: obj) =
            match o with
            | :? AssemblyReceiver as x -> serverRef.CompareTo(x.ServerRef)
            | _ -> -1
     
/// FsPickler for Akka's IActorRefs       
let actorRefPickler (system: ExtendedActorSystem) : Pickler<IActorRef> =
    let toSurrogate (ref: IActorRef) : string = 
        Akka.Serialization.Serialization.SerializedActorPath(ref)
    let fromSurrogate (surrogate: string) : IActorRef = 
        system.Provider.ResolveActorRef(surrogate)
    Pickler.wrap fromSurrogate toSurrogate Pickler.string

/// Akka's Serializer for types which needs info about Vagabond-loaded types
type VagabondSerializer<'T>(system, vm: VagabondManager, id) = 
    inherit Akka.Serialization.Serializer(system)
    override __.IncludeManifest = false
    override __.Identifier = id
    override x.ToBinary(o) = 
        let piriMsg = o :?> 'T
        use stream = new MemoryStream()
        vm.Serializer.Serialize(stream, piriMsg)
        stream.ToArray()
    override x.FromBinary(bin, t) =
        use stream = new MemoryStream(bin)
        upcast vm.Serializer.Deserialize<'T>(stream)