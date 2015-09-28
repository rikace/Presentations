namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("FSFastWeb")>]
[<assembly: AssemblyProductAttribute("FSFastWeb")>]
[<assembly: AssemblyDescriptionAttribute("FSharp samples in the Web")>]
[<assembly: AssemblyVersionAttribute("1.0")>]
[<assembly: AssemblyFileVersionAttribute("1.0")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "1.0"
