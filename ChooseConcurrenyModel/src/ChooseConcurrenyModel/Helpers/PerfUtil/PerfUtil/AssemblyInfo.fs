namespace System
open System.Reflection

[<assembly: AssemblyTitleAttribute("PerfUtil")>]
[<assembly: AssemblyProductAttribute("PerfUtil")>]
[<assembly: AssemblyDescriptionAttribute("A simple F# utility for testing performance")>]
[<assembly: AssemblyVersionAttribute("0.1.8")>]
[<assembly: AssemblyFileVersionAttribute("0.1.8")>]
do ()

module internal AssemblyVersionInformation =
    let [<Literal>] Version = "0.1.8"
