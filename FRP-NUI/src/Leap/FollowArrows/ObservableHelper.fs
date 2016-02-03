namespace FollowArrowRX

open Microsoft.FSharp.Control
open System.Reactive
open System.Reactive.Subjects
open System.Reactive.Linq
open System
open System.Threading
open RxFsharp

[<AutoOpen>]
module ObservableHelper =
    type Observable with
        static member keepState = Microsoft.FSharp.Control.Observable.scan