namespace FRPFSharp

type Listener = {
    Unlisten: unit -> unit
}
with 
    member this.Append(other) =
        { Unlisten = fun () -> this.Unlisten(); other.Unlisten() }