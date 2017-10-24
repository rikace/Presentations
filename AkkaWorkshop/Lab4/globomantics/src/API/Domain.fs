module API.Domain

open System
open Akkling

type UserId = int
type VideoId = int

[<Flags>]
type Genre =
    | Crime = 1
    | Drama = 2
    | Action = 4
    | Biography = 8
    | History = 16
    | Adventure = 32
    | Fantasy = 64
    | Western = 128
    | Romance = 256
    | Mystery = 512
    | SciFi = 1024

type Video =
    { Id: VideoId
      Title: string
      Genres: Genre
      RunningTime: TimeSpan
      Rating: double }
