// Decorator Pattern
[<AbstractClassAttribute>]
type ComputerParts() =
    abstract member Description :unit -> unit

type Computer() =
    inherit ComputerParts()
    override O.Description() = printfn "I'm a Computer with"

type CDROM( c :ComputerParts ) =
    inherit ComputerParts()
    override O.Description() = c.Description(); printfn ", CDROM"

type Mouse( c :ComputerParts ) =
    inherit ComputerParts()
    override O.Description() = c.Description(); printfn ", Mouse"

type Keyboard( c :ComputerParts ) =
    inherit ComputerParts()
    override O.Description() = c.Description(); printfn ", Keyboard"

let mutable computer = Computer() :> ComputerParts
computer <- Mouse( computer )
computer <- CDROM( computer )
computer <- Keyboard( computer )

computer.Description()