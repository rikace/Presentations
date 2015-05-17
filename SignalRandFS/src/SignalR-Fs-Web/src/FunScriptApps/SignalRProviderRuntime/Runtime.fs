namespace SignalRProviderRuntime

open FunScript

[<ReflectedDefinition>]
type JsonObject = 
    [<JSEmitInlineAttribute("({})")>]
    static member Create() = failwith "Funscript emit" : obj
    
    [<JSEmitInlineAttribute("(({0})[{1}] = {2})")>]
    static member Set<'x> (ob: obj) (propertyName: string) (value: 'x) = failwith "FunScript emit" : unit

    [<JSEmitInlineAttribute("(({0})[{1}]")>]
    static member Get (ob: obj) (propertyName: string) = failwith "FunScript emit" : 'x


[<ReflectedDefinition>]
type HubUtil =
    [<JSEmitInline("({0}.client = {1})")>]
    static member RegisterClientProxy (h : obj) (x: obj) =failwith "FunScript emit" : unit

type ClientHubAttribute() = 
    inherit System.Attribute()

    member val HubName = "ClientHub" with get,set