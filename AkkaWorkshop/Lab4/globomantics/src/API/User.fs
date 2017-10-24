module API.User

open System
open API.Domain
open API.DataAccess
open Akkling
open Akkling.Persistence

type UserState =
    { UserId: UserId
      SeenVideos: VideoId list }
    static member Zero = { UserId = -1; SeenVideos = [] }

type UserCommand =
    | Login of userId:UserId
    | RecommendVideo of userId:UserId
    | MarkVideoAsWatched of userId:UserId * videoId:VideoId

type UserEvent =
    // event
    | VideoWatched of videoId:VideoId

/// message extractor for cluster sharding. It takes UserCommand,
/// extracts an UserId from it and uses it to route it to correct
/// shard and user. Shard count determines max number of shards.
let route shardCount (msg: obj) =
    match downcast msg with
    | Login userId
    | RecommendVideo userId
    | MarkVideoAsWatched(userId, _) ->
        let shardId = userId % shardCount |> string
        let entityId = userId |> string
        (shardId, entityId, msg)

/// Props used to etablish persistent user actor.
let props videoStore = propsPersist(fun ctx ->
    // behavior in which user is already logged in
    let rec loggedIn state: Effect<obj> = actor {
        let! (msg: obj) = ctx.Receive()
        match msg with
        | SnapshotOffer snapshot -> return! loggedIn snapshot
        | :? UserEvent as e ->
            match e with
            | VideoWatched vid ->
                if not <| ctx.IsRecovering()
                then
                    // if event doesn't come from recovering, but as result of Persist
                    // then we print a message to the console
                    printfn "User %d watched video: %A" state.UserId vid
                return! loggedIn { state with SeenVideos = vid::state.SeenVideos }
        | :? UserCommand as c ->
            match c with
            | Login userId ->
                return! loggedIn { state with UserId = userId }
            | RecommendVideo _ ->
                /// <<! is forward operator used to send a message to other actor without changing the Sender
                videoStore <<! UnseenVideosRequest(state.UserId, state.SeenVideos)
                return! loggedIn state
            | MarkVideoAsWatched(_, vid) ->
                // we persist an UserEvent as result of this command
                // once persisted, it will call current behavior with ctx.IsRecovering() flag off
                return Persist (upcast VideoWatched vid)
        | _ -> return Unhandled }
    /// actor starts in unlogged state - it needs to receive Login message to change its behavior
    and unlogged state = actor {
        let! (msg: obj) = ctx.Receive()
        match msg with
        | SnapshotOffer snapshot -> return! unlogged snapshot
        | :? UserEvent as e ->
            match e with
            | VideoWatched vid ->
                // in this behavior every event is recovering, since MarkVideoAsWatched command that produces
                // this event is handled only in loggedIn behavior
                return! unlogged { state with SeenVideos = vid::state.SeenVideos }
        | :? UserCommand as c ->
            match c with
            | Login userId ->
                return! loggedIn { state with UserId = userId }
            | other ->
                printfn "Cannot %A. User not logged in." other
                return Unhandled
        | _ -> return Unhandled
    }
    unlogged UserState.Zero)