namespace  AsyncTask

open System 
open System.Diagnostics 
open System.Threading

[<AutoOpen>]
 module Task =
    let getData(uri:string) =
        Async.StartAsTask <|
        async { let request = System.Net.WebRequest.Create uri
                use! response = request.AsyncGetResponse()
                return [    use stream = response.GetResponseStream()
                            use reader = new System.IO.StreamReader(stream)
                            while not reader.EndOfStream
                                do yield reader.ReadLine() ] }