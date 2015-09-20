namespace Easj360FSharp

open System.IO

module FastStreamCopy =

        type RequestGate(n:int) =
            let sem = new System.Threading.Semaphore(n, n)
            member x.Acquire(?timeout) =
                async { let! ok = Async.AwaitWaitHandle(sem, ?millisecondsTimeout=timeout)
                    if (ok) then
                        return
                            { new System.IDisposable with
                                member x.Dispose() =
                                    sem.Release() |> ignore }
                    else
                        return! failwith "Couldn't acquire Gate" }  

        type Microsoft.FSharp.Control.Async with
                 static member Parallel2(actionOne, actionTwo) =
                    async { 
                            let! resOne = Async.StartChild actionOne
                            let! resTwo = Async.StartChild actionTwo
                            let! completeOne = resOne
                            return! resTwo
                          }

        type SuperFastFileCopy(concurrency:int) =
            let gate = new RequestGate(concurrency)

            member public x.Start(sourceFile:string, detsinationFile:string) =
                Async.Start(
                    async { use! wait = gate.Acquire()
                            use inStream = new FileStream(sourceFile, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)     
                            use outStream = new FileStream(detsinationFile, System.IO.FileMode.Create, System.IO.FileAccess.Write, System.IO.FileShare.None, 4096, System.IO.FileOptions.Asynchronous ||| System.IO.FileOptions.SequentialScan)     
                            let sizeBuffer = 4096
                            let buffers = Array.init 2 (fun _ -> Array.zeroCreate<byte> sizeBuffer) 
                            let indexBuffer = 0   
                            let! bytesReadFirstRound = inStream.AsyncRead(buffers.[indexBuffer],0,sizeBuffer)
                            let rec copyAsync(bytesRead:int, arrayIndex:int) = async {
                                match bytesRead with
                                | 0 -> return ()
                                | m -> let readArrayIndex = arrayIndex
                                       let writeIndex = arrayIndex ^^^ 1
                                       let! bytesRead = Async.Parallel2(outStream.AsyncWrite(buffers.[readArrayIndex], 0, m), inStream.AsyncRead(buffers.[writeIndex],0,sizeBuffer))
                                       return! copyAsync(bytesRead, writeIndex)
                                       }
                            do! copyAsync(bytesReadFirstRound, indexBuffer)                
                            }
                        )

            member private x.CopyStream(inStream:System.IO.Stream, outStream:System.IO.Stream) =
                     Async.Start(
                        async { use! wait = gate.Acquire()                 
                                let sizeBuffer = 4096
                                let buffers = Array.init 2 (fun _ -> Array.zeroCreate<byte> sizeBuffer) 
                                let indexBuffer = 0   
                                let! bytesReadFirstRound = inStream.AsyncRead(buffers.[indexBuffer],0,sizeBuffer)
                                let rec copyAsync(bytesRead:int, arrayIndex:int) = async {
                                    match bytesRead with
                                    | 0 -> return ()
                                    | m -> let readArrayIndex = arrayIndex
                                           let writeIndex = arrayIndex ^^^ 1
                                           let! bytesRead = Async.Parallel2(outStream.AsyncWrite(buffers.[readArrayIndex], 0, m), inStream.AsyncRead(buffers.[writeIndex],0,sizeBuffer))
                                           return! copyAsync(bytesRead, writeIndex)
                                           }
                                do! copyAsync(bytesReadFirstRound, indexBuffer)                
                             })