module Façade

// define Applicant record
type Applicant = { Name : string }

// library to perform various checks
[<AutoOpen>]
module SubOperationModule = 
    let checkCriminalRecord (applicant) = 
        printfn "checking %s criminal record..." applicant.Name
        true

    let checkPastEmployment (applicant) = 
        printfn "checking %s past employment..." applicant.Name
        true

    let securityClearance (applicant, securityLevel) = 
        printfn "security clearance for %s ..." applicant.Name
        true

// façade function to perform the background check
let isBackgroundCheckPassed(applicant, securityLevel) = 
    checkCriminalRecord applicant
    && checkPastEmployment applicant
    && securityClearance(applicant, securityLevel)

// create an applicant 
let jenny = { Name = "Jenny" }

// print out background check result
if isBackgroundCheckPassed(jenny, 2) then printfn "%s passed background check" jenny.Name
else printfn "%s failed background check" jenny.Name
