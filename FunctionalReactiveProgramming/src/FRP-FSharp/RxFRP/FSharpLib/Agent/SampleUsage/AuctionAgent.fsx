#load "AgentSystem.fs"
open AgentSystem.LAgent
open System
open System.Threading

type AuctionClientState = int * int
and AuctionAgentState = bool * int * AsyncAgent<AuctionReply,unit>
and AuctionSellerState = unit
and AuctionMessage =
  | Offer of int * AsyncAgent<AuctionReply, AuctionClientState> // Make a bid
  | Inquire of AsyncAgent<AuctionReply, AuctionClientState>     // Check the status
and AuctionReply =
  | StartBidding
  | Status of int * DateTime // Asked sum and expiration
  | BestOffer                // Ours is the best offer
  | BeatenOffer of int       // Yours is beaten by another offer
  | AuctionConcluded of      // Auction concluded
      AsyncAgent<AuctionReply, AuctionSellerState> * AsyncAgent<AuctionReply, AuctionClientState>
  | AuctionFailed            // Failed without any bids
  | AuctionOver              // Bidding is closed
  
let timeToShutdown = 3000
let bidIncrement = 10 

let auctionAgent seller minBid closing =
    let agent = spawnAgent (fun (msg:AuctionMessage) (isConcluded, maxBid, maxBidder) ->
                            match msg with
                            | Offer (_, client) when isConcluded ->
                                client <-- AuctionOver
                                (isConcluded, maxBid, maxBidder)
                            | Offer(bid, client) when not(isConcluded) ->
                                if bid >= maxBid + bidIncrement then
                                    if maxBid >= minBid then maxBidder <-- BeatenOffer bid                  
                                    client <-- BestOffer
                                    (isConcluded, bid, client)
                                else
                                    client <-- BeatenOffer maxBid
                                    (isConcluded, maxBid, maxBidder)
                            | Inquire client    ->
                                client <-- Status(maxBid, closing)
                                (isConcluded, maxBid, maxBidder))
                            (false, (minBid - bidIncrement), Unchecked.defaultof<AsyncAgent<_, _>>)                             
                                
    agent <-! SetTimeoutHandler(
                (closing - DateTime.Now).Milliseconds,
                (fun (isConcluded: bool, maxBid, maxBidder) ->
                    if maxBid >= minBid then
                      let reply = AuctionConcluded(seller, maxBidder)
                      maxBidder <-- reply
                      seller <-- reply
                    else seller <-- AuctionFailed
                    agent <-! SetTimeoutHandler(
                        timeToShutdown,
                        (fun (_:bool, _:int,_:AsyncAgent<_,_>) -> StopProcessing))
                    ContinueProcessing (true, maxBid, maxBidder)))   
    agent            
        
module Auction =
  let random = new Random()
  
  let minBid = 100
  let closing = DateTime.Now.AddMilliseconds 10000.
  
  let seller = spawnWorker (fun (msg:AuctionReply) -> ())
  let auction = auctionAgent seller minBid closing
  
  let client i increment top = 
    let name = sprintf "Client %i" i
    let log msg = Console.WriteLine("{0}: {1}", name, msg)
    
    let rec c = spawnAgent (
                    fun msg (max: int, current) ->
                        let processBid (aMax, aCurrent) =
                            if aMax >= top then
                                log "too high for me"
                                (aMax, aCurrent)
                            elif aCurrent < aMax then
                                  let aCurrent = aMax + increment
                                  Thread.Sleep (1 + random.Next 1000)
                                  auction <-- Offer(aCurrent, c)
                                  (aMax, aCurrent)
                            else (aMax, aCurrent)                       
                        match msg with
                        | StartBidding      ->
                            auction <-- Inquire c
                            (max, current)
                        | Status(maxBid,_)  ->
                            log <| sprintf "status(%d)" maxBid
                            let s = processBid (maxBid, current)
                            c <-! SetTimeoutHandler( timeToShutdown, (fun _ -> StopProcessing) )
                            s
                        | BestOffer ->
                            log <| sprintf "bestOffer(%d)" current
                            processBid(max, current)
                        | BeatenOffer maxBid ->
                            log <| sprintf "beatenOffer(%d)" maxBid
                            processBid(maxBid, current)
                        | AuctionConcluded(seller, maxBidder) ->
                            log "auctionConcluded"
                            c <-! Stop
                            (max, current)
                        | AuctionOver ->
                            log "auctionOver"
                            c <-! Stop
                            (max, current))
                     (0,0)
    c
                            
  
open Auction

(client 1 20 200) <-- StartBidding
(client 2 10 300) <-- StartBidding
(client 3 30 150) <-- StartBidding
Console.ReadLine() |> ignore  
    