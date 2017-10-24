namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("K-Means.FSharp")>]
[<assembly: AssemblyProductAttribute("FunctionalConcurrencyNET")>]
[<assembly: AssemblyDescriptionAttribute("Source code for the book Functional Concurrency .NET")>]
[<assembly: AssemblyVersionAttribute("1.0")>]
[<assembly: AssemblyFileVersionAttribute("1.0")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "1.0"
