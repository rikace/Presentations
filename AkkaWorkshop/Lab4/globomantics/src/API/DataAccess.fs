module API.DataAccess

open System
open API.Domain

let private minutes = TimeSpan.FromMinutes
let private inmemStore: Video list = [
    { Id = 0; Title = "The Shawshank Redemption"; Genres = Genre.Crime|||Genre.Drama; RunningTime = minutes 142.; Rating = 9.2 }
    { Id = 1; Title = "The Godfather"; Genres = Genre.Crime|||Genre.Drama; RunningTime = minutes 175.; Rating = 9.2 }
    { Id = 2; Title = "The Godfather: Part II"; Genres = Genre.Crime|||Genre.Drama; RunningTime = minutes 202.; Rating = 9. }
    { Id = 3; Title = "The Dark Knight"; Genres = Genre.Action|||Genre.Crime|||Genre.Drama; RunningTime = minutes 152.; Rating = 8.9 }
    { Id = 4; Title = "Pulp Fiction"; Genres = Genre.Crime|||Genre.Drama; RunningTime = minutes 154.; Rating = 8.9 }
    { Id = 5; Title = "Schindler's List"; Genres = Genre.Drama|||Genre.Biography|||Genre.History; RunningTime = minutes 195.; Rating = 8.9 }
    { Id = 6; Title = "12 Angry Men"; Genres = Genre.Crime|||Genre.Drama; RunningTime = minutes 96.; Rating = 8.9 }
    { Id = 7; Title = "The Lord of the Rings: The Return of the King"; Genres = Genre.Adventure|||Genre.Drama|||Genre.Fantasy; RunningTime = minutes 201.; Rating = 8.9 }
    { Id = 8; Title = "The Good, the Bad and the Ugly"; Genres = Genre.Western; RunningTime = minutes 148.; Rating = 8.9 }
    { Id = 9; Title = "Fight Club"; Genres = Genre.Drama; RunningTime = minutes 139.; Rating = 8.8 }
    { Id = 10; Title = "The Lord of the Rings: The Fellowship of the Ring"; Genres = Genre.Adventure|||Genre.Drama|||Genre.Fantasy; RunningTime = minutes 178.; Rating = 8.8 }
    { Id = 11; Title = "Star Wars: Episode V - The Empire Strikes Back"; Genres = Genre.Adventure|||Genre.Action|||Genre.Fantasy; RunningTime = minutes 124.; Rating = 8.7 }
    { Id = 12; Title = "Forrest Gump"; Genres = Genre.Romance; RunningTime = minutes 142.; Rating = 8.7 }
    { Id = 13; Title = "Inception"; Genres = Genre.Action|||Genre.Mystery|||Genre.SciFi; RunningTime = minutes 148.; Rating = 8.7 }
    { Id = 14; Title = "One Flew Over the Cuckoo's Nest"; Genres = Genre.Drama; RunningTime = minutes 133.; Rating = 8.7 }
    { Id = 15; Title = "The Lord of the Rings: The Two Towers"; Genres = Genre.Adventure|||Genre.Drama|||Genre.Fantasy; RunningTime = minutes 179.; Rating = 8.7 } ]

let private getUnseenVideos watchedIds = async {
    return inmemStore
        |> List.filter (fun vid -> watchedIds |> List.exists ((=)vid.Id))
}

open Akkling

type FetcherMsg =
    | UnseenVideosRequest of userId:UserId * seenIds:VideoId list

let fetcherProps = props(fun ctx ->
    let rec loop () = actor {
        let! msg = ctx.Receive()
        match msg with
        | UnseenVideosRequest(userId, vids) ->
            async {
                let! unseen = getUnseenVideos vids
                let result =
                    unseen
                    |> List.sortByDescending (fun v -> v.Rating)
                    |> List.take 5
                return Recommendation(userId, result)
            } |!> ctx.Sender()
        return! loop ()
    }
    loop ())