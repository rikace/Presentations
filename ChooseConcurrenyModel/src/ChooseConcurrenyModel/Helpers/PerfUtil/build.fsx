// --------------------------------------------------------------------------------------
// FAKE build script 
// --------------------------------------------------------------------------------------

#I "packages/FAKE/tools/"
#r @"packages/FAKE/tools/FakeLib.dll"

open Fake 
open Fake.Git
open Fake.AssemblyInfoFile
open Fake.ReleaseNotesHelper
open System

// The name of the project 
// (used by attributes in AssemblyInfo, name of a NuGet package and directory in 'src')
let project = "PerfUtil"

// Short summary of the project
// (used as description in AssemblyInfo and as a short summary for NuGet package)
let summary = "A simple F# utility for testing performance"

// Longer description of the project
// (used as a description for NuGet package; line breaks are automatically cleaned up)
let description = """
  A simple F# utility for testing performance """
// List of author names (for NuGet package)
let authors = [ "Eirik Tsarpalis" ]
// Tags for your project (for NuGet package)
let tags = "F#, performance, tests, unit, tests"

// File system information 
// (<solutionFile>.sln is built during the building process)
let solutionFile  = "PerfUtil"

//// Pattern specifying assemblies to be tested using NUnit
//let testAssemblies = "tests/**/bin/Release/*Tests*.dll"

// Git configuration (used for publishing documentation in gh-pages branch)
// The profile where the project is posted 
let gitHome = "https://github.com/eiriktsarpalis"
// The name of the project on GitHub
let gitName = "PerfUtil"

// --------------------------------------------------------------------------------------
// END TODO: The rest of the file includes standard build steps 
// --------------------------------------------------------------------------------------

// Read additional information from the release notes document
Environment.CurrentDirectory <- __SOURCE_DIRECTORY__
let release = parseReleaseNotes (IO.File.ReadAllLines "RELEASE_NOTES.md")

// Generate assembly info files with the right version & up-to-date information
let mkAssemblyInfo project =
  let fileName = project + "/AssemblyInfo.fs"
  CreateFSharpAssemblyInfo fileName
    [ 
        Attribute.Title project
        Attribute.Product project
        Attribute.Description summary
        Attribute.Version release.AssemblyVersion
        Attribute.FileVersion release.AssemblyVersion 
    ] 

Target "AssemblyInfo" (fun _ ->
    mkAssemblyInfo "PerfUtil"
    mkAssemblyInfo "PerfUtil.NUnit"
)

// --------------------------------------------------------------------------------------
// Clean build results & restore NuGet packages

Target "RestorePackages" RestorePackages

Target "Clean" (fun _ ->
    CleanDirs ["bin"; "temp"]
)

Target "CleanDocs" (fun _ ->
    CleanDirs ["docs/output"]
)

// --------------------------------------------------------------------------------------
// Build library & test project

Target "Build" (fun _ ->
    !! (solutionFile + "*.sln")
    |> MSBuildRelease "" "Rebuild"
    |> ignore
)

//// --------------------------------------------------------------------------------------
//// Run the unit tests using test runner
//
//Target "RunTests" (fun _ ->
//    !! testAssemblies 
//    |> NUnit (fun p ->
//        { p with
//            DisableShadowCopy = true
//            TimeOut = TimeSpan.FromMinutes 20.
//            OutputFile = "TestResults.xml" })
//)

// --------------------------------------------------------------------------------------
// Build a NuGet package

Target "NuGet -- PerfUtil" (fun _ ->
    NuGet (fun p -> 
        { p with   
            Authors = authors
            Project = "PerfUtil"
            Summary = summary
            Description = description
            Version = release.NugetVersion
            ReleaseNotes = String.Join(Environment.NewLine, release.Notes)
            Tags = tags
            OutputPath = "bin"
            AccessKey = getBuildParamOrDefault "nugetkey" ""
            Publish = hasBuildParam "nugetkey" })
        "nuget/PerfUtil.nuspec"
)

Target "NuGet -- PerfUtil.NUnit" (fun _ ->
    NuGet (fun p -> 
        { p with   
            Authors = authors
            Project = "PerfUtil.NUnit"
            Summary = "NUnit extensions for PerfUtil"
            Description = "NUnit extensions for PerfUtil"
            Version = release.NugetVersion
            ReleaseNotes = String.Join(Environment.NewLine, release.Notes)
            Tags = tags
            OutputPath = "bin"
            AccessKey = getBuildParamOrDefault "nugetkey" ""
            Publish = hasBuildParam "nugetkey" 
            Dependencies = [("PerfUtil", release.NugetVersion) ; ("NUnit", "2.6.3")] 
        })
        "nuget/PerfUtil.NUnit.nuspec"
)

//// --------------------------------------------------------------------------------------
//// Generate the documentation
//
//Target "GenerateDocs" (fun _ ->
//    executeFSIWithArgs "docs/tools" "generate.fsx" ["--define:RELEASE"] [] |> ignore
//)
//
//// --------------------------------------------------------------------------------------
//// Release Scripts
//
//Target "ReleaseDocs" (fun _ ->
//    let tempDocsDir = "temp/gh-pages"
//    CleanDir tempDocsDir
//    Repository.cloneSingleBranch "" (gitHome + "/" + gitName + ".git") "gh-pages" tempDocsDir
//
//    fullclean tempDocsDir
//    CopyRecursive "docs/output" tempDocsDir true |> tracefn "%A"
//    StageAll tempDocsDir
//    Commit tempDocsDir (sprintf "Update generated documentation for version %s" release.NugetVersion)
//    Branches.push tempDocsDir
//)
//
//

// --------------------------------------------------------------------------------------
// Run all targets by default. Invoke 'build <Target>' to override

Target "Release" DoNothing
Target "All" DoNothing

"Clean"
  ==> "RestorePackages"
  ==> "AssemblyInfo"
  ==> "Build"
//  ==> "RunTests"
  ==> "All"

"All" 
//  ==> "CleanDocs"
//  ==> "GenerateDocs"
//  ==> "ReleaseDocs"
  ==> "NuGet -- PerfUtil"
  ==> "NuGet -- PerfUtil.NUnit"
  ==> "Release"

RunTargetOrDefault "Release"