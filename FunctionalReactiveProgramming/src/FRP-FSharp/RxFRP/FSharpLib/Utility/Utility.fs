namespace Easj360FSharp

module Utility =

let (|Default|) defaultValue input =
    defaultArg input defaultValue


