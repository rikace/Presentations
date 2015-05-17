module MonkeyHub


open System
open System.Text
open EkonBenefits.FSharp.Dynamic
open Microsoft.AspNet.SignalR
open Microsoft.AspNet.SignalR.Hubs
open FSharp.Collections.ParallelSeq

type Agent<'a> = MailboxProcessor<'a>

type RandomAgentMessage =
    | GetRandomNextDouble of AsyncReplyChannel<float>
    | GetRandomNextRange of int * int * AsyncReplyChannel<int>
    
type RandomThreadSafe() =
        let rnd = Random()
        let o = Object()
        member x.NextDouble() =
            lock o (fun () -> rnd.NextDouble())

        member x.Next(min, max) =
            lock o (fun () -> rnd.Next(min, max))



//type RandomThreadSafe() = 
//        let agent = Agent<RandomAgentMessage>.Start(fun inbox -> 
//                        let rnd = new Random()
//                        let rec loop count = async {
//                            let! msg = inbox.Receive()
//                            match msg with
//                            | GetRandomNextDouble(reply) -> let n = rnd.NextDouble() 
//                                                            reply.Reply( n )
//                            | GetRandomNextRange(min, max, reply) -> let n = rnd.Next(min,max)
//                                                                     reply.Reply( n )
//                            return! loop ( count + 1)   }
//                        loop 0)
//        
//        member x.NextDouble() = agent.PostAndAsyncReply(fun ch -> GetRandomNextDouble(ch))
//        member x.NextRange(min, max) = agent.PostAndAsyncReply(fun ch -> GetRandomNextRange(min, max, ch))

type GeneticAlghorithmSettings = {PopulationSize:int; MutationProbability:float; CrossoverProbability:float} 
                                 static member GetDefaultValues() = { PopulationSize = 200; MutationProbability = 0.04; CrossoverProbability = 0.87}

let computeFitness (text:string) (targetText:string) = lazy 
                          if (not <| String.IsNullOrEmpty(text)) && (not <| String.IsNullOrEmpty(targetText)) then
                              [0.. targetText.Length - 1] |> List.filter(fun i -> targetText.[i] <> text.[i]) |> List.length
                          else Int32.MaxValue 

[<StructAttribute>]
type TextMatchGenome(text:string, targetText:string) = 
        member x.Text = text
        member x.TargetText = targetText
        member x.Fitness = computeFitness text targetText 
         
type PopulationMessage =
        | CreatePulation of AsyncReplyChannel<TextMatchGenome option>
  
type TextMacthGeneticAlghorithm(targetText:string, ?settings:GeneticAlghorithmSettings) =
        let settings = defaultArg settings (GeneticAlghorithmSettings.GetDefaultValues())
        let randomThreadSafe = RandomThreadSafe()
        static let validChars = seq {   yield char 10
                                        yield char 13
                                        yield! [ 2 .. 96] |> List.mapi(fun n _ -> char (n + 32)) } |> Seq.toArray
        
        let createRandomGenome() = 
                let text =     [0..targetText.Length - 1] 
                               |> PSeq.map(fun _ -> randomThreadSafe.Next(0, validChars.Length))
                               |> PSeq.fold(fun (acc:StringBuilder) index -> acc.Append(validChars.[index])) (new StringBuilder(targetText.Length))
                TextMatchGenome(text.ToString(),targetText) 
        
        let crossover (p1:TextMatchGenome) (p2:TextMatchGenome) = 
            let crossoverPoint = randomThreadSafe.Next(1, p1.Text.Length)
            let child1 = TextMatchGenome((p1.Text.Substring(0, crossoverPoint) + p2.Text.Substring(crossoverPoint)), targetText)
            let child2 = TextMatchGenome((p2.Text.Substring(0, crossoverPoint) + p1.Text.Substring(crossoverPoint)), targetText)
            (child1, child2) 

        let mutate(genome:TextMatchGenome) = 
            let sb = new StringBuilder(genome.Text)
            let e1 = randomThreadSafe.Next(0, genome.Text.Length)
            let e2 = randomThreadSafe.Next(0, validChars.Length)
            sb.[e1] <- validChars.[e2]
            TextMatchGenome(sb.ToString(), genome.TargetText) 

        let createChildren(parent1:TextMatchGenome) (parent2:TextMatchGenome) = 
            let nextDouble = randomThreadSafe.NextDouble()
            let child1, child2 =    if nextDouble < settings.CrossoverProbability then
                                        (crossover parent1 parent2)                                       
                                    else parent1, parent2 
            let child1' = let nextDouble = randomThreadSafe.NextDouble()
                          if nextDouble < settings.MutationProbability then 
                                mutate child1
                          else child1 
            
            let child2' = let nextDouble = randomThreadSafe.NextDouble()
                          if nextDouble < settings.MutationProbability then 
                            mutate child2
                          else child2 
            [| child1'; child2' |]
            
        let findRandomHighQuanlityParent (currentPopulation:TextMatchGenome array) (sumOfMaxMinusFitness:int64) (maxFitness:int) =  
            let nextDouble = randomThreadSafe.NextDouble()
            let value = int64 (nextDouble * float(sumOfMaxMinusFitness))
            let rec findRandomHighQuanlityParent index accValue = 
                let maxMinusFitnes = int64 ( maxFitness - currentPopulation.[index].Fitness.Value )
                match accValue < maxMinusFitnes with
                | true  -> currentPopulation.[index]
                | false -> findRandomHighQuanlityParent (index + 1) (accValue - maxMinusFitnes)
            findRandomHighQuanlityParent 0 value 
            
        let createRandomPopulation() = 
                    [| 0 .. settings.PopulationSize |]
                    |> Array.Parallel.map(fun _ -> createRandomGenome()) 
            
        let createNextGeneration(currentPopulation:TextMatchGenome array) =  
                let maxFitness = 
                    let maxFitness = currentPopulation |> PSeq.maxBy(fun t -> t.Fitness.Value) 
                    maxFitness.Fitness.Value + 1
                let sumOfMaxMinusFitness = 
                    currentPopulation 
                        |> Array.Parallel.map(fun t -> int64(maxFitness - t.Fitness.Value))
                        |> PSeq.sum
                let nextGeneration = 
                        [|0 .. settings.PopulationSize /2 |]
                        |> Array.Parallel.map(fun _ -> let child1 =  (findRandomHighQuanlityParent(currentPopulation) sumOfMaxMinusFitness maxFitness)
                                                       let child2 =  (findRandomHighQuanlityParent(currentPopulation) sumOfMaxMinusFitness maxFitness)                                                                
                                                       createChildren(child1) (child2) )
                        |> Array.Parallel.collect id
                nextGeneration 
                     
        let currentPopulationAgent = 
            Agent<PopulationMessage>.Start(fun inbox ->                                   
                    let rec loop (currentPopulation:TextMatchGenome array) = async {
                            let! msg = inbox.Receive()                            
                            match msg with
                            | CreatePulation(reply) when currentPopulation.Length = 0 -> 
                                            let population = createRandomPopulation()                                          
                                            reply.Reply( Some(population.[0]) )
                                            return! loop population
                            | CreatePulation(reply) -> 
                                            let currentBestGenoma = currentPopulation.[0]
                                            let population = createNextGeneration(currentPopulation)    
                                            if  population.[0].Fitness.Value < currentBestGenoma.Fitness.Value then                                             
                                                reply.Reply( Some(population.[0]) )
                                            else reply.Reply(None)
                                            return! loop population
                        }
                    loop [||])
        
        member x.MoveNext() =
            currentPopulationAgent.PostAndAsyncReply(fun ch -> CreatePulation(ch))


[<HubName("monkeys")>]
type MonkeyHub() as this= 
    inherit Hub()

    let cancellation = new System.Threading.CancellationTokenSource()

    member x.StartTyping(targetText:string):unit = 
        let settings = GeneticAlghorithmSettings.GetDefaultValues()
        let token = cancellation.Token
        let ga = TextMacthGeneticAlghorithm(targetText, settings)
        
        this.Clients.All?started(targetText)

        let startedAt = DateTime.Now

        let rec moveNextGeneration generation = async {
            let! getNextGeneration = ga.MoveNext()
            match getNextGeneration with
            | Some(g) ->    let generationsPerSecond = generation / (max 1 (int (DateTime.Now - startedAt).TotalSeconds))
                            this.Clients.All?updateProgress(g.Text, generation, generationsPerSecond)
                            if(g.Fitness.Value = 0) then
                                this.Clients.All?complete()
                                return ()
                            else return! moveNextGeneration (generation + 1)
            | None -> return! moveNextGeneration (generation + 1) }
        Async.Start(moveNextGeneration 1, cancellation.Token)

    member x.StopTyping() = 
            cancellation.Cancel()

