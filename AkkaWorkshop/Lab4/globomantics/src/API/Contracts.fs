[<AutoOpen>]
module API.Messages

open Akkling
open API.Domain

// this is a place where public contract between API frontend and backend logic is placed

type Response =
    | Recommendation of userId:UserId * recommended:Video list