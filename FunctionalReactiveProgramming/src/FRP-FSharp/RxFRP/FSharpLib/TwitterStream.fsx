//----------------------------------------------------------------------------
//
// Copyright (c) 2010-2011 Microsoft Corporation. 
//
// This source code is subject to terms and conditions of the Apache License, Version 2.0. 
// By using this source code in any fashion, you are agreeing to be bound 
// by the terms of the Apache License, Version 2.0.
//
// You must not remove this notice, or any other, from this software.
//----------------------------------------------------------------------------

namespace global

#r "System.Web.dll"
#r "System.Windows.Forms.dll"
#r "System.Xml.dll"
#r "System.Xml.Linq.dll"
#r "System.Runtime.Serialization.dll"

open System
open System.IO
open System.Net 
open System.Text 
open System.Runtime.Serialization
open System.Runtime.Serialization.Json
open System.Collections.Generic
open System.Globalization
open System.Net.Security
open System.Web
open System.Threading
                
/// A component which listens to tweets in the background and raises an 
/// event each time a tweet is observed
type TwitterStreamSample(userName:string, password:string) = 
    //let filteredWords = System.IO.File.ReadAllLines(__SOURCE_DIRECTORY__ + @"\words.txt")
    let mutable event = new Event<string>()   
    let credentials = NetworkCredential(userName, password)
    let streamSampleUrl = "http://stream.twitter.com/1/statuses/sample.json?delimited=length"
    let listener (ctxt:System.Threading.SynchronizationContext) = 
        async { let req = WebRequest.Create(streamSampleUrl, Credentials=credentials) 
                use resp = req.GetResponse()
                //use! resp = req.AsyncGetResponse()
                use stream = resp.GetResponseStream()
                use reader = new StreamReader(stream)
                while not reader.EndOfStream do 
                    let size = reader.ReadLine() 
                    if not (String.IsNullOrEmpty size) then 
                        let size = int size
                        let buffer = Array.zeroCreate size
                        reader.ReadBlock(buffer,0,size)  |> ignore
                        let text = String buffer
                        ctxt.Post((fun _ -> event.Trigger text), null) }

    let ctxt = System.Threading.SynchronizationContext.Current
    do Async.Start(listener ctxt)

    member this.StopListening() = event <- new Event<_>()

    /// Feeds the XML one tweet at a time
    member this.NewTweet = event.Publish


[<AutoOpen>]
module Tweet = 

    //-----------------------------
    // Tweet Parsing

    open System
    open System.Xml.Linq
    open System.Web
    open System.Globalization

    /// The results of the parsed tweet
    [<DataContract>]
    type UserStatus = 
        { [<field: DataMember(Name="screen_name") >]
          mutable UserName : string;
          [<field: DataMember(Name="friends_count") >]
          mutable FriendsCount : int;
          [<field: DataMember(Name="followers_count") >]
          mutable FollowersCount : int;
          [<field: DataMember(Name="created_at") >]
          mutable JoinDate : string
        }

    [<DataContract>]
    type Tweet = 
        { [<field: DataMember(Name="user") >]
          mutable User : UserStatus 
          [<field: DataMember(Name="created_at") >]
          mutable StatusDate : string
          [<field: DataMember(Name="text") >]
          mutable Status: string;  }


    /// Object from Json 
    let unjson<'T> (jsonString:string)  : 'T =  
            use ms = new MemoryStream(System.Text.ASCIIEncoding.Default.GetBytes(jsonString)) 
            let obj = (new DataContractJsonSerializer(typeof<'T>)).ReadObject(ms) 
            obj :?> 'T


    /// Attempt to parse a tweet
    let parseTweet (tweet: string) = 
        try 
           let t = unjson<Tweet> tweet 
           match box t.User with 
           | null -> None
           | _ -> Some t
        with _ -> 
           None

