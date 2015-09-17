// 45-50
// Same as async example. Computes length of info on homepage. 
module ScalingAgents

open System.Net

type Agent<'T> = MailboxProcessor<'T>

let urlList = [ ("Microsoft.com", "http://www.microsoft.com/");
                ("MSDN", "http://msdn.microsoft.com/");
                ("Google", "http://www.google.com") ]

let processingAgent() = Agent<string * string>.Start(fun inbox ->
                        async { while true do
                                let! name,url = inbox.Receive()
                                let uri = new System.Uri(url)
                                let webClient = new WebClient()
                                let! html = webClient.AsyncDownloadString(uri)
                                printfn "Read %d characters for %s" html.Length name })

let scalingAgent : Agent<(string * string) list> = Agent.Start(fun inbox -> 
                                    async { while true do 
                                            let! msg = inbox.Receive()
                                            msg
                                            |> List.iter (fun x -> 
                                                            let newAgent = processingAgent()
                                                            newAgent.Post x )})

scalingAgent.Post urlList


