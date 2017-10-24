module SimpleTypeProvider

open FSharp.Data

[<Literal>]
let simpleJson = "{ 
    \"id\": 1, 
    \"first-name\": \"Avi\", 
    \"last-name\": \"Avni\", 
    \"Birthday\": \"1986-11-07\", 
    \"data\": [1, 2, 3, 4] 
}"
type Example = JsonProvider<simpleJson>
let example = Example.GetSample()

let id = example.Id
let firstName = example.FirstName
let lastName = example.LastName
let birthday = example.Birthday
let data = example.Data