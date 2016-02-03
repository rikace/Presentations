
#r "System.Web.dll"
#r "System.Windows.Forms.dll"
#r "System.Xml.dll"
 
open System
open System.Globalization
open System.IO
open System.Net
open System.Web
open System.Threading
open Microsoft.FSharp.Control.WebExtensions

/// A component which listens to tweets in the background and raises an
/// event each time a tweet is observed
type TwitterStreamSample(userName:string, password:string) =
 
    let tweetEvent = new Event<_>()  
    let streamSampleUrl = "http://stream.twitter.com/1.1/statuses/sample.xml?delimited=length"
    //let streamSampleUrl = "https://stream.twitter.com/1.1/statuses/filter.json?delimited=length&track=twitterapi:" //https://stream.twitter.com/1.1/statuses/sample.json"
    
    /// The cancellation condition
    let mutable group = new CancellationTokenSource()
 
    /// Start listening to a stream of tweets
    member this.StartListening() =
                                                                 /// The background process

        // Capture the synchronization context to allow us to raise events back on the GUI thread
       // let syncContext = SynchronizationContext.Current
 
        let listener =
            async { let credentials = NetworkCredential(userName, password)
                    let req = WebRequest.Create(streamSampleUrl, Credentials=credentials)
                    use! resp = req.AsyncGetResponse()
                    use stream = resp.GetResponseStream()
                    use reader = new StreamReader(stream)
                    let atEnd = reader.EndOfStream
                    let rec loop() =
                        async {
                            printfn "start"
                            let atEnd = reader.EndOfStream
                            if not atEnd then
                                let sizeLine = reader.ReadLine()
                                if String.IsNullOrEmpty sizeLine then return! loop() else
                                let size = int sizeLine
                                let buffer = Array.zeroCreate size
                                let _numRead = reader.ReadBlock(buffer,0,size) 
                                let text = new System.String(buffer)
                                printfn "%s" text
                                tweetEvent.Trigger text
                                return! loop()
                        }
                    return! loop() }
 
        Async.Start(listener, group.Token)
 
    /// Stop listening to a stream of tweets
    member this.StopListening() =
        group.Cancel();
        group <- new CancellationTokenSource()
 
    /// Raised when the XML for a tweet arrives
    member this.NewTweet = tweetEvent.Publish

/////////
#load @"C:\Git\Easj360Git\FSharpLib\show.fs";;

let userName = "Rikace"
let password = ""
let twitterStream = new TwitterStreamSample(userName, password)
twitterStream.NewTweet.Add (fun s -> printfn "%A" s) 
twitterStream.StartListening()

twitterStream.StopListening()

show <| "cio"

//////////////////
#r "System.Xml.dll"
#r "System.Xml.Linq.dll"
open System.Xml
open System.Xml.Linq 
let xn (s:string) = XName.op_Implicit s
/// The results of the parsed tweet
type UserStatus =
    { UserName : string
      ProfileImage : string
      Status : string
      StatusDate : DateTime }
/// Attempt to parse a tweet
let parseTweet (xml: string) =  
    let document = XDocument.Parse xml
    let node = document.Root
    if node.Element(xn "user") <> null then
        Some { UserName     = node.Element(xn "user").Element(xn "screen_name").Value;
               ProfileImage = node.Element(xn "user").Element(xn "profile_image_url").Value;
               Status       = node.Element(xn "text").Value       |> HttpUtility.HtmlDecode;
               StatusDate   = node.Element(xn "created_at").Value |> (fun msg ->
                                   DateTime.ParseExact(msg, "ddd MMM dd HH:mm:ss +0000 yyyy",
                                                       CultureInfo.CurrentCulture)); }
    else
        None


twitterStream.NewTweet
   |> Event.choose parseTweet
   |> Event.add (fun s -> printfn "%A" s)
 
twitterStream.StartListening()

//////////////////
let addToMultiMap key x multiMap =
   let prev = match Map.tryFind key multiMap with None -> [] | Some v -> v
   Map.add x.UserName (x::prev) multiMap
 
/// An event which triggers on every 'n' triggers of the input event
let every n (ev:IEvent<_>) =
   let out = new Event<_>()
   let count = ref 0
   ev.Add (fun arg -> incr count; if !count % n = 0 then out.Trigger arg)
   out.Publish
 
twitterStream.NewTweet
   |> Event.choose parseTweet
   // Build up the table of tweets indexed by user
   |> Event.scan (fun z x -> addToMultiMap x.UserName x z) Map.empty
   // Take every 20’th index
   |> every 20
   // Listen and display the average of #tweets/user
   |> Event.add (fun s ->
        let avg = s |> Seq.averageBy (fun (KeyValue(_,d)) -> float d.Length)
        printfn "#users = %d, avg tweets = %g" s.Count avg)
 
twitterStream.StartListening()

