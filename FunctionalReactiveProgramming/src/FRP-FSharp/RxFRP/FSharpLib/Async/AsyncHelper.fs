namespace Easj360FSharp

open Microsoft.FSharp.Control
open Microsoft.FSharp.Control.WebExtensions
open Microsoft.FSharp.Core
open System.Text
open System.IO

[<AutoOpen>]
module AsyncHelpers =

    type Microsoft.FSharp.Control.Async with
        static member Parallel2 (a1, a2) =
            async { let! job1 = Async.StartChild a1
                    let! job2 = Async.StartChild a2
                    let! res1 = job1
                    let! res2 = job2
                    return res1, res2 }


    type System.IO.StreamReader with 
        member reader.AsyncReadAllLines () = 
            async { return [ while not reader.EndOfStream do 
                                yield reader.ReadLine()] }
    
    type System.Net.WebRequest with
        member req.AsyncGetRequestStream() = 
            Async.FromBeginEnd(req.BeginGetRequestStream, req.EndGetRequestStream)
           


        member req.AsyncWriteContent (content:string) = 
            async { let bytes = Encoding.UTF8.GetBytes content
                    req.ContentLength <- int64 bytes.Length
                    use! stream = req.AsyncGetRequestStream()
                    do! stream.AsyncWrite bytes  }

        member req.WriteContent (content:string) = 
            let bytes = Encoding.UTF8.GetBytes content
            req.ContentLength <- int64 bytes.Length
            use stream = req.GetRequestStream()
            stream.Write(bytes,0,bytes.Length)

        member req.AsyncReadResponse () = 
            async { use! response = req.AsyncGetResponse()
                    use responseStream = response.GetResponseStream()
                    use reader = new StreamReader(responseStream)
                    return! reader.AsyncReadToEnd() }

        member req.AsyncReadResponseLines () = 
            async { use! response = req.AsyncGetResponse()
                  
                    return [use stream = response.GetResponseStream()
                            use reader = new StreamReader(stream)
                            while not reader.EndOfStream do 
                                yield reader.ReadLine()] }

        member req.ReadResponse () = 
            use response = req.GetResponse()
            use responseStream = response.GetResponseStream()
            use reader = new StreamReader(responseStream)
            reader.ReadToEnd()