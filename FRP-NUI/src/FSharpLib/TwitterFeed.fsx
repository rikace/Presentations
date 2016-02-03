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
// F# Twitter Feed Sample using Event processing
//

#load "show.fs"
#load "events.fs"
#load "stats.fs"
#load "TwitterStream.fsx"

//----------------------------------------------
//
let userName : string = "TRikace"
let password : string = "Jocker74!"

let twitterStream = 
    new TwitterStreamSample(userName, password)

twitterStream.NewTweet
   |> Event.add show 


twitterStream.StopListening(); show "ready..."


//-----------------------------
// Tweet Parsing


twitterStream.NewTweet 
    |> Event.map parseTweet
    |> Event.add show

twitterStream.StopListening(); show "ready..."

//-----------------------------
// Tweet Parsing (words)


twitterStream.NewTweet 
    |> Event.choose parseTweet
    |> Event.map (fun tweet -> tweet.User.UserName, tweet.Status)
    |> Event.add show

twitterStream.StopListening(); show "ready..."



//-----------------------------
// Word analysis

let words (s:string) = 
   s.Split([|' ' |], System.StringSplitOptions.RemoveEmptyEntries)

words "All the pretty horses" |> show



//-----------------------------
// Word counting


twitterStream.NewTweet
   // Parse the tweets
   |> Event.choose parseTweet
   // Get the words in the status
   |> Event.map (fun x -> words x.Status)
   // Add the words to an incremental histogram
   |> Event.histogram
   // Visualize every 3 tweets
   |> Event.every 3
   // For each event, find the top 50 entries
   |> Event.map (Histogram.top 50)
   // Show
   |> Event.add show


twitterStream.StopListening(); show "ready..."



//-----------------------------
// User Counting


twitterStream.NewTweet
   // Parse the tweets
   |> Event.choose parseTweet
   // Incrementally index by user name
   |> Event.indexBy (fun tweet -> tweet.User.UserName)
   // Visualize every 10 tweets
   |> Event.every 10
   // Find the current count and average/user
   |> Event.map (fun s -> 
         let avg = s |> Seq.averageBy (fun (KeyValue(_,d)) -> float d.Length)
         sprintf "#users = %d, avg tweets = %g" s.Count avg)
   // Show
   |> Event.add show

twitterStream.StopListening(); show "ready..."



//-----------------------------
// Tweet indexing


twitterStream.NewTweet 
    |> Event.choose parseTweet
    |> Event.scan (fun z x -> MultiMap.add x.User.UserName x.Status z) MultiMap.empty
    // Visualize every 10 tweets
    |> Event.every 10
    // Show
    |> Event.add show

twitterStream.StopListening(); show "ready..."








