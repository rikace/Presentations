#light
module PopolateArray
open System
open System.IO
open System.Net

let getValue p =
    p + 1

let CreateArray2 (value:int32) =
    let (arr2:int32[,]) = Array2D.create value value (int32 0)
    for i=0 to value-1 do 
        for j=0 to value-1 do
            arr2.[i,j] <- getValue i
        done
    done
    arr2
    
let CreateArray (value:int32) =
    let (arr1:int32[]) = Array.create value (int32 0)
    for i=0 to value-1 do 
            arr1.[i] <- getValue i
    done

let Operation2 (value:Int32, slice:int32) =
    let chunk = value / slice
    let lastChunk = value - chunk * slice 
    let arr1 = CreateArray2(chunk)
    let arr2 = CreateArray2(lastChunk)    
    arr1.GetLength(0)
    
let Operation (value:Int32, slice:int32) =
    let chunk = value / slice
    let lastChunk = value - chunk * slice 
    let arr1 = CreateArray(chunk)
    let arr2 = CreateArray(lastChunk)    
    arr1
    
type System.Net.WebRequest with
     member x.GetResponseAsync() =
     Async.FromBeginEnd(x.BeginGetResponse, x.EndGetResponse)


let OperationAsync (value:Int32, slice:int32) =
    let chunk = value / slice
    let lastChunk = value - chunk * slice 
    let arr1 = async { let result = Operation(value, slice)
                       return (value:int32) }
    let arr2 = async { let result = Operation(value, slice)
                       return (value:int32) }
    let arr3 = async { let result = Operation(value, slice)
                       return (value:int32) }
    let arr4 = async { let result = Operation(value, slice)
                       return (value:int32) }
    Async.RunSynchronously(Async.Parallel [ arr1; arr2; arr3; arr4; ])

let AsyncHttp(url:string) =
        async {  do printfn "Created web request for %s" url
                 let req = WebRequest.Create(url)
                 do printfn "Getting response for %s" url
                 let! rsp = req.AsyncGetResponse()
                 do printfn "Reading response for %s" url
                 use stream = rsp.GetResponseStream()
                 use reader = new System.IO.StreamReader(stream)
                 return! reader.AsyncReadToEnd()
               }

type Company = { Name:string; Size:int; } 

let companySizeComparer = 
  { new System.Collections.Generic.IComparer<Company> with 
      override x.Compare(c1, c2) = 
        c1.Size.CompareTo(c2.Size) 
  } 

