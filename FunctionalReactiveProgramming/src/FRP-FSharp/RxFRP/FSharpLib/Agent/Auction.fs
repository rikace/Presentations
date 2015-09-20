namespace Eaasj360

// F# port of auction.scala: http://lampsvn.epfl.ch/svn-repos/scala/scala/trunk/docs/examples/actors/auction.scala
open System
open System.Threading

module Auction = 

    let (<--) (m:'msg MailboxProcessor) x = m.Post x
    let unSome (Option.Some(x)) = x

    type AuctionMessage =
      | Offer of int * AuctionReply MailboxProcessor // Make a bid
      | Inquire of AuctionReply MailboxProcessor     // Check the status
    and AuctionReply =
      | Status of int * DateTime // Asked sum and expiration
      | BestOffer                // Ours is the best offer
      | BeatenOffer of int       // Yours is beaten by another offer
      | AuctionConcluded of      // Auction concluded
          AuctionReply MailboxProcessor * AuctionReply MailboxProcessor
      | AuctionFailed            // Failed without any bids
      | AuctionOver              // Bidding is closed

    let timeToShutdown = 3000
    let bidIncrement = 10

    let auctionAgent seller minBid closing =
      new MailboxProcessor<AuctionMessage> (fun inbox ->
        let rec loop maxBid maxBidder =
          async { let! msg = inbox.TryReceive((closing - DateTime.Now).Milliseconds)
                  match msg with
                    | Some ( Offer(bid, client) ) ->
                        if bid >= maxBid + bidIncrement then
                          if maxBid >= minBid then unSome maxBidder <-- BeatenOffer bid                  
                          client <-- BestOffer
                          return! loop bid (Some client)
                        else
                          client <-- BeatenOffer maxBid
                          return! loop maxBid maxBidder
                      
                    | Some ( Inquire client ) ->
                        client <-- Status(maxBid, closing)
                        return! loop maxBid maxBidder
                    
                    | None ->
                        if maxBid >= minBid then
                          let reply = AuctionConcluded(seller, unSome maxBidder)
                          unSome maxBidder <-- reply
                          seller <-- reply
                        else seller <-- AuctionFailed
                        let! msg' = inbox.TryReceive timeToShutdown
                        match msg' with
                        | Some ( Offer (_, client) ) -> 
                            client <-- AuctionOver
                            return! loop maxBid maxBidder
                        | None -> return ()         
                }
        loop (minBid - bidIncrement) None)   

    module Auction =
      let random = new Random()
  
      let minBid = 100
      let closing = DateTime.Now.AddMilliseconds 10000.
  
      let seller = new MailboxProcessor<AuctionReply>(fun inbox ->
        let rec loop() =
          async { let! _ = inbox.Receive() 
                  return! loop()}
        loop())
      let auction = auctionAgent seller minBid closing
  
      let client i increment top = 
        let name = sprintf "Client %i" i
        let log msg = Console.WriteLine("{0}: {1}", name, msg)
    
        new MailboxProcessor<AuctionReply>(fun inbox ->
    
          let rec startAuction() =
            async { log "started"
                    auction <-- Inquire inbox
                    let! curMsg = inbox.Receive()
                    match curMsg with
                    | Status(maxBid,_) ->
                        log <| sprintf "status(%d)" maxBid
                        return! loop 0 maxBid }
          and loop current max =
            async { if max >= top then log "too high for me"
                    
                    let current' =
                      if current < max then
                        let current' = max + increment
                        Thread.Sleep (1 + random.Next 1000)
                        auction <-- Offer(current', inbox)
                        current'
                      else current
                
                    let! msg = inbox.TryReceive timeToShutdown
                    match msg with
                    | Some BestOffer -> 
                        log <| sprintf "bestOffer(%d)" current'
                        return! loop current' max
                    | Some (BeatenOffer maxBid) ->
                        log <| sprintf "beatenOffer(%d)" maxBid
                        return! loop current' maxBid
                    | Some ( AuctionConcluded(seller, maxBidder) ) ->
                        log "auctionConcluded"; return ()
                    | Some AuctionOver ->
                        log "auctionOver"; return ()
                    | None -> return () }
          startAuction())

    open Auction

    seller.Start()
    auction.Start()
    (client 1 20 200).Start()
    (client 2 10 300).Start()
    (client 3 30 150).Start()
    Console.ReadLine() |> ignore