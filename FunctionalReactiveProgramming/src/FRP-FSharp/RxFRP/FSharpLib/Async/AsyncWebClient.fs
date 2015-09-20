namespace Easj360FSharp

module AsyncWebClient =

        open System.IO
        open System.Net
        open System


        type WebClient with
          member this.AsyncDownloadData (address:Uri) : Async<byte array> =
            async { let userToken = new obj()
                    let cancelAction() = this.CancelAsync()  
                    this.DownloadDataAsync(address, userToken)

                    let rec loop() = 
                      async { let! args = Async.AwaitEvent(this.DownloadDataCompleted,cancelAction=this.CancelAsync)
                              if args.UserState <> userToken then return! loop()
                              else
                                let result = 
                                  if args.Cancelled then 
                                    AsyncCanceled (new OperationCanceledException())
                                  elif args.Error <> null then  AsyncException args.Error
                                  else  AsyncOk args.Result 
                                return! AsyncResult.Commit(result) }
                    return! loop() }

          member this.AsyncDownloadFile (address:Uri, filename:string) : Async<unit> =
            async { let userToken = new obj()
                    let cancelAction() = this.CancelAsync()  
                    this.DownloadFileAsync(address, filename, userToken)

                    let rec loop() = 
                      async { let! args = Async.AwaitEvent(this.DownloadFileCompleted,cancelAction=this.CancelAsync)
                              if args.UserState <> userToken then return! loop()
                              else
                                let result = 
                                  if args.Cancelled then 
                                    AsyncCanceled (new OperationCanceledException())
                                  elif args.Error <> null then  AsyncException args.Error
                                  else  AsyncOk ()
                                return! AsyncResult.Commit(result) }
                    return! loop() }

          member this.AsyncOpenRead (address:Uri) : Async<Stream> =
            async { let userToken = new obj()
                    let cancelAction() = this.CancelAsync()  
                    this.OpenReadAsync(address, userToken)

                    let rec loop() = 
                      async { let! args = Async.AwaitEvent(this.OpenReadCompleted,cancelAction=this.CancelAsync)
                              if args.UserState <> userToken then return! loop()
                              else
                                let result = 
                                  if args.Cancelled then 
                                    AsyncCanceled (new OperationCanceledException())
                                  elif args.Error <> null then  AsyncException args.Error
                                  else  AsyncOk args.Result
                                return! AsyncResult.Commit(result) }
                    return! loop() }

          member this.AsyncOpenWrite (address:Uri, ?methodName:string) : Async<unit> =
            async { let userToken = new obj()
                    let cancelAction() = this.CancelAsync()  
                    let methodName = defaultArg methodName null
                    this.OpenWriteAsync(address, methodName, userToken)

                    let rec loop() = 
                      async { let! args = Async.AwaitEvent(this.OpenWriteCompleted,cancelAction=this.CancelAsync)
                              if args.UserState <> userToken then return! loop()
                              else
                                let result = 
                                  if args.Cancelled then 
                                    AsyncCanceled (new OperationCanceledException())
                                  elif args.Error <> null then  AsyncException args.Error
                                  else  AsyncOk ()
                                return! AsyncResult.Commit(result) }
                    return! loop() }
            
            
