open System
open Akkling
open API
open Akkling.Cluster.Sharding.ClusterSharding

let printerActor = function
    | Recommendation(userId, unseen) ->
        Console.ForegroundColor <- ConsoleColor.Green
        printfn "Recommended videos for user %d: %A" userId unseen
        Console.ResetColor()
        |> ignored

[<EntryPoint>]
let main argv =
    let config =
        Configuration.load().
            WithFallback(Akka.Cluster.Tools.Singleton.ClusterSingletonManager.DefaultConfig()).
            WithFallback(Akka.Cluster.Sharding.ClusterSharding.DefaultConfig()).
            WithFallback(Akka.Persistence.Sqlite.SqlitePersistence.DefaultConfiguration())

    use system = System.create "globomantics" <| config
    let rand = Random()

    // Cluster sharding proxy is used to communicate with cluster shard regions. This requires:
    // 1. routing strategy - the same as the one used in spawnSharded function
    // 2. actor system
    // 3. type name of shards used to group all shards in all cluster nodes
    // 4. optional name of the role for nodes, on which cluster sharding is expected to be found
    //    cluster proxy has role [api], while nodes with cluster sharding active have [api] role
    //
    // we don't need actor props here, as this node won't allocate any sharded actors, just
    // communicate with them on other nodes
    let proxy = spawnShardedProxy (User.route 100) system "user" (Some "api") |> retype
    let printer = spawnAnonymous system <| props(actorOf printerActor)

    printfn "Enter to start emitting messages..."
    Console.ReadLine() |> ignore

    let userCount = 11
    system.Scheduler.Advanced.ScheduleRepeatedly(TimeSpan.FromSeconds 2., TimeSpan.FromSeconds 4., fun () ->
        // pick a random user, send a login request to it and then a recommendation request
        let userId = rand.Next userCount
        proxy <! User.Login userId
        proxy.Tell(User.RecommendVideo userId, printer))

    system.Scheduler.Advanced.ScheduleRepeatedly(TimeSpan.FromSeconds 1., TimeSpan.FromSeconds 2., fun () ->
        proxy <! User.MarkVideoAsWatched (rand.Next userCount, rand.Next 16))

    system.WhenTerminated.Wait()
    0 // return an integer exit code
