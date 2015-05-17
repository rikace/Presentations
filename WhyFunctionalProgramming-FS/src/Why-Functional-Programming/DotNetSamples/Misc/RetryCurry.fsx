let print = function
            | Some(s) -> printfn "%s" s
            | None -> printfn "No result"

let retry (f: unit -> 'b) =
    let rec retry count  =
        match count with
        | 0 -> None
        | n ->  try
                    let result = f()
                    Some(result)
                with
                | _ ->  retry (n - 1)
    retry 3 
    


let msft = "http://microsoft.com"
let client = new System.Net.WebClient()
let download = (fun () -> client.DownloadString(msft))
let result = retry download


let filePath = __SOURCE_DIRECTORY__ + "\TextFile.txt"
let read() = System.IO.File.ReadAllText(filePath)
let text = (read) |> retry 
print text

let read' path = (fun () -> System.IO.File.ReadAllText path) 
let text' = retry (read'(filePath))
print text'


