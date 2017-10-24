open System

// --------------------------------------------------------
// Functions in practice - Hole in the middle pattern
// --------------------------------------------------------

// EXAMPLE: Hole in the middle pattern

// Builds a simple IDicitionary with roles
let roles = dict [ ("Tomas", "Admin"); ("Phil", "User") ]

let writeData user =
  match roles.TryGetValue(user) with
  | true, role -> 
      // Perform some work depending on the role
      if role = "Admin" then printfn "Welcome admin!"
      else printfn "Welcome user!"
  | _ -> printfn "LOG: unknown user '%s'" user

writeData "Tomas"
writeData "Phil"
writeData "Evil Tomas"

// --------------------------------------------------------------
// TODO: Use function values to avoid duplicate code in the 
// following two functions that work with 'Service' object

#load "Service.fs"
open HolePattern

let printSkillsMatterTitle() =
  let svc = new Service("http://skillsmatter.com")
  try
    // Do the actual work using the service
    printfn "%s" (svc.GetTitle())
  with
  | :? ApplicationException as e ->
    printfn "Error: %s" e.Message


let printSkillsMatterLength() =
  let svc = new Service("http://skillsmatter.com")
  try
    // Do the actual work using the service
    printfn "%d" (svc.GetLength())
  with
  | :? ApplicationException as e ->
    printfn "Error: %s" e.Message

printSkillsMatterTitle()
printSkillsMatterLength()