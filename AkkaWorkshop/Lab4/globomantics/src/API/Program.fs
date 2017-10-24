open System
open API
open API.Domain
open Akkling
open Akkling.Cluster
open Akkling.Cluster.Sharding.ClusterSharding

[<EntryPoint>]
let main argv = 
    let config = 
        Configuration.load().
            WithFallback(Akka.Cluster.Tools.Singleton.ClusterSingletonManager.DefaultConfig()).
            WithFallback(Akka.Cluster.Sharding.ClusterSharding.DefaultConfig()).
            WithFallback(Akka.Persistence.Sqlite.SqlitePersistence.DefaultConfiguration())

    use system = System.create "globomantics" <| config
    let videoStore = spawn system "videoStore" DataAccess.fetcherProps

    // in order to use cluster sharding, we need to initialize shard region, which takes several parameters:
    // 1. routing method used to extract shardId and entityId from user messages - in cluster sharding actors 
    //    are identified by shardId/entityId and not by their actor paths
    // 2. actor system
    // 3. type name of shards used to group all shards in all cluster nodes
    // 4. Props used to initialize actor - in cluster sharding actors are not created explicitly. Instead they 
    //    are created ad-hoc on one of the cluster nodes having sharding region for that actor type, once they 
    //    get a message that defines a shard-id/entity-id for an actor.
    let userRegion = spawnSharded (User.route 100) system "user" (User.props videoStore)
    system.WhenTerminated.Wait()
    0 // return an integer exit code
