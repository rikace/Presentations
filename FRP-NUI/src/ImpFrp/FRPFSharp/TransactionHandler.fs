namespace FRPFSharp

open System

type TransactionHandler<'a> = {
    mutable Run: Atomically -> 'a -> unit
    mutable CurrentListener: Listener option
}
with
    static member New(run: Action<Atomically, 'a>) =
        { CurrentListener = None; Run = fun t a -> run.Invoke(t, a) }
    interface ICloneable with
        member this.Clone() =
            upcast { CurrentListener = None; Run = this.Run }

    interface IDisposable with
        member this.Dispose() =
            match this.CurrentListener with
            | Some l -> l.Unlisten()
            | None -> ()
