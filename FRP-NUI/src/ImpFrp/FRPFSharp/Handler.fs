namespace FRPFSharp

type Handler<'a> = {
    mutable Fired: bool
    mutable Run: 'a -> unit
}
with 
    static member New(f: System.Action<'a>) : Handler<'a> =
        { Fired = false; Run = fun a -> f.Invoke(a) }