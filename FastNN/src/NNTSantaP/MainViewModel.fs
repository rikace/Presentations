namespace ViewModels

open System
open System.Threading
open FSharp.ViewModule
open FSharp.Charting
open FSharp.Charting.ChartTypes
open System.Windows.Forms.Integration
open FSharp.Control.Reactive

[<AutoOpen>]
module Utilities =
    let [<Literal>] rangeLen = 1000.
    let rand = new Random(int DateTime.Now.Ticks)

    type RandMessage =
        | GetMap of int * AsyncReplyChannel<(float * float)[]>
        | SetMap of int * AsyncReplyChannel<(float * float)[]>

    let mapAgent =
        MailboxProcessor.Start(fun inbox ->
            let initMap n = Array.init n (fun _ -> rand.NextDouble()*rangeLen, rand.NextDouble()*rangeLen)
            let rec loop (map:(float*float)[]) =
                async {
                    let! msg = inbox.Receive()
                    match msg with
                    | GetMap(n, reply) ->
                        let map =
                            match map with
                            | x when n = x.Length -> map
                            | _ -> initMap n
                        reply.Reply(map)
                        return! loop map
                    | SetMap(n, reply) ->
                        let map = initMap n
                        reply.Reply(map)
                        return! loop map
                }
            loop (initMap 0))



// Step .1
// lets start by defining the model representing the Neuron
// the neuron in this case has only 2 properties
//      - weights as an array of floats. (technically it could be a point with coordinate X Y, for simplicity we are using an array)
//      - output as a single float, which is the result of the computation
//
//  the Neuron can be defined as a RecordType in F#
//      for example, a Record type for a Person is
//          type Person = {firstName:string; lastName:string }
//      to create a record type simply
//              let person = {firstName="Riccardo";  lastName="Terrell" }
//      the F# type inference will do the rest


// [ YOUR CODE GOES HERE! ]
type Neuron =
    { weights : float[]
      output  : float }
    member this.inputsCount = this.weights.Length
    member this.item n = this.weights.[n]

// should be helpful to have 2 functions to access the properties
// of a Neuron instance
// for example to access the length of the weights use this function (uncomment)
//    member this.inputsCount = this.weights.Length


//    then create a function that access an item of the Neuron weight array
//    with the following initial definition

//    member this.item n = [ code here ]



// Step .2
// create 2 functions helpers for the Neuron
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix )>]
module Neuron =
    let create weight' =
        // the create function returns a default Neuron RecordType with
        // output zero and weight as the weight' passed
        // [ YOUR CODE GOES HERE! ]
        { output = 0.
          weights = weight' }

    // create a Neuron with random weight
    let createRandom (inputs : int) =
        create <| Array.init (max 1 inputs) (fun _ -> rand.NextDouble() * rangeLen)


    let compute (neuron : Neuron) (input : float[]) =
        // create a function that return an array of float
        // each value is the sum of the pairs between the value of
        // the item from the Neuron Weight and the input item having same indexes
        //      for example
        //      let arr1 = [|1;2|]
        //      let arr2 = [|3;4|]
        //      the result is arr3 = [|4;6|]
        //
        // F# has a great support for sequence manipulation
        //      for example Seq.map   Seq.filter ...
        //      in this case, beside a for-loop you could try to
        //      Seq.zip    and    Seq.sumBy


        // [ YOUR CODE GOES HERE! ]
        neuron.weights
        |> Seq.zip input
        |> Seq.sumBy (fun (a,b) -> abs(a-b))



// Step .3
// taking as example the Neuron record-type, lets repeat the same approach
// to define a Layer type
// a Layer has 2 properties, a neurons property to access an array of Neurons
// and an array of float for the output

// in addition add 2 instance function for the Layer Record-Type
//      1 - get the counts on Neurons
//      2 - get a neuron given an index n

/// A layer represents a collection of neurons
// [ YOUR CODE GOES HERE! ]


type Layer =
    { neurons : Neuron[]
      output  : float[] }
    member this.neuronsCount = this.neurons.Length
    member this.item n = this.neurons.[n]

[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Layer =

// Step .4
// define a function to create a Layer type
// the arguments are the integer for count of Neurons,
// and a function used to initialize each Neuron with signature (int -> Neuron)
//
//      in F# is handy the Array module.
//      for example, to initialize an Array with a give function you could use the Array.init function
//                   that takes as argument the count of Neurons
//                   and a function to create each item  (int -> (int -> 'a)) where 'a is a Neuron type
//                   for example, to create an array of string you can use the following code
//                                for simplicity the string is the result of converting each array-index number to string (but you get the idea :) )//
//
//                                Array.int 100 (fun x -> string x)
//
//                                in F# everything is a function, the 'string' function take a generic type 'a and converted into a string
//                                thus, you could rewrite the previous code as follow
//
//                                Array.init 100 (string)
//
//                  to create an array with default values for the output there several options, try to found out what the Array module can do

    let create (neuronsCount:int) (ctorFunc:int -> Neuron) =  // [ YOUR CODE GOES HERE! ]
        {
          neurons = Array.init neuronsCount ctorFunc
          output = Array.zeroCreate<float> neuronsCount
        }

    // Create Layer with shape of bubble with `R` radius center in `c` point
    let createBubble neuronsCount (c:float[]) R =
        let neuronsCount = max 1 neuronsCount
        let delta = Math.PI * 2.0 / (float neuronsCount)
        let initFunc i =
            let alpha = float(i) * delta
            let x' = c.[0] + R * Math.Cos(alpha)
            let y' = c.[1] + R * Math.Sin(alpha)

            // add the function to create a Neuron early defined
            // -  the argument is an array of float, which in this case are just the coordinate of the Neuron x y
            // [ YOUR CODE GOES HERE! ]
            Neuron.create [|x'; y'|]
        create neuronsCount initFunc

    // Create Layer with random neuron location
    let createRandom neuronsCount inputsCount  =
        create neuronsCount (fun i -> Neuron.createRandom inputsCount)

    /// Compute output vector of the layer
    let compute (inputs : float array) (layer : Layer) : Layer=
        let neuronsCount = layer.neuronsCount
        let output = Array.init neuronsCount (fun i -> Neuron.compute layer.neurons.[i] inputs)
        { layer with output = output }



// Step .5 Create Single hidden layer called Network, which is just an alias for the Layer :)
// [ YOUR CODE GOES HERE! ]
type Network = Layer // Single hidden layer


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module Network =
    let createBubble = Layer.createBubble
    let zero = Layer.createRandom 0 2

    let createRandom inputsCount neuronsCount =
        Layer.createRandom neuronsCount inputsCount

    let compute (network : Network) (input : float array) =
        network |> Layer.compute input


// Step .5 Create Single hidden layer called Network, which is just an alias for the Layer :)
// [ YOUR CODE GOES HERE! ]

//  complete the findBestOutput function
//  from the output array in the Network type
//  found the best value. However, is not that simple :)
//  you should create a collection composed by the tuple of each value of the output array
//  and its index in the array.
//  for example. an array of [|10;30|] would become [|(10,0); (30,1)|]  where 0 and 1 are the index
//  of the position of a give value in the array
//  the found the min value between the tuples and return its index value

    let findBestOutput (network : Network) =
    // [ YOUR CODE GOES HERE! ]

        network.output
        |> Seq.mapi (fun i o -> (o,i))  
        |> Seq.minBy id
        |> snd

// Definition of the Elastic Network
type ElasticNetworkLearning =
    { learningRate : float
      learningRadius : float
      squaredRadius : float
      distance : float array
      network : Network }


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module NetworkLearning =
    let create (network : Network) =
        let neuronsCount = network.neuronsCount
        let delta = Math.PI * 2.0 / (float neuronsCount)

        let rec initDistance i alpha acc =
            match i with
            | n when n < neuronsCount ->
                let x = 0.5 * Math.Cos(alpha) - 0.5
                let y = 0.5 * Math.Sin(alpha)
                initDistance (i + 1) (alpha + delta) ((x * x + y * y)::acc)
            | _ -> acc |> List.toArray
        // initial arbitrary values
        { learningRate = 0.1
          learningRadius = 0.5
          squaredRadius = 98.
          distance = initDistance 0 delta []
          network = network }

    let setLearningRate learningRate (learning : ElasticNetworkLearning) =
        { learning with learningRate = max 0. (min 1. learningRate) }

    let setLearningRadius learningRadius (learning : ElasticNetworkLearning) =
        let learningRadius = max 0. (min 1. learningRadius)
        { learning with learningRadius = learningRadius
                        squaredRadius = 2. * learningRadius * learningRadius }

    let compute (learning : ElasticNetworkLearning) (input : float array) =
        let learningRate = learning.learningRate
        let network = Network.compute learning.network input
        let bestNeuronId = Network.findBestOutput network
        let layer = network // Just single hidden layer in network


// Step .6 return a new ElasticNetworkLearning with same values as the (learning : ElasticNetworkLearning)
// argument but with an update network value
        for j = 0 to layer.neuronsCount - 1 do
        // [ YOUR CODE GOES HERE! ]
        // take a neuron from the layer value using the helper function to access a neuron
        // with a given index, in this case j

        //  calculate the factor value , which is the 'exp' of the negative number of the
        //  distance value of the ElasticNetworkLearning (you can access this value with the learning.distance property)
        //          -> having as index the 'abs' value of the current index j minus the best-NeuronId
        //  divide by the squaredRadius value of the ElasticNetworkLearning
        //
        //  for example
        //      let factor = exp (- learning.distance.[abs .... more code here
        //
        // then update each neuron weights
        //      the new neuron weight is the sum of its own current value
        //      plus the learningRate
        //      plus the difference between the value take it from the input array and the neuron weight
        //          both having same index in the array (good solution is a for loop and using the same index to access both array)
        //          this difference  (input.[index] - neuron.... ) multiplied by the factor value early computed
        //
        //  return a new (learning : ElasticNetworkLearning) value with the updated network value
        //      in F# Record-Type are immutable. You can create a new Record-Type from an existing one with different
        //      property(ies) using the `with` keyword
        //          for example, to create a new record type from an existing `person` one but only with the `age` property different
        //                      let newPerson = { person with age = 42 }
        //      help: this is how mutate a value in an array in F#
        //                  neuron.weghts.[index] <- new value here
        // [ YOUR CODE GOES HERE! ]
            let neuron = layer.item j
            let factor = exp (-learning.distance.[abs (j - bestNeuronId)] / learning.squaredRadius)
            for i = 0 to neuron.inputsCount - 1 do
                let e = (input.[i] - neuron.item i) * factor
                neuron.weights.[i] <- neuron.weights.[i] + (e + learningRate)
        { learning with network = network }



type KNIES =
    { learningBubble : int
      learningRate : float
      network : Network }


[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module KNIES =
    let create (network : Network) =
        { learningBubble = 1 + network.neurons.Length / 5
          learningRate = 0.1
          network = network }

    let setLearningBubble learningBubble (learning : KNIES) =
        { learning with learningBubble = max 1 (min (learning.network.neuronsCount / 3) learningBubble) }

    let setLearningRate learningRate (learning : KNIES) =
        { learning with learningRate = max 0. (min 1. learningRate) }

    let compute (learning : KNIES) (input : float array) =
        let network = Network.compute learning.network input
        let j' = Network.findBestOutput network
        let layer = network // Just single hidden layer in network

        let dist j = let dist' = abs (j - j')
                     min dist' (network.neuronsCount - dist')
        let isInBubble j = (dist j) < learning.learningBubble

        // Attraction phase (move closest neuron to the city)
        let bubbleSize = 2 * learning.learningBubble - 1
        let attractedLayer =
            let bubbleSize = float(learning.learningBubble)
            Layer.create layer.neuronsCount (fun j ->
                let neuron = layer.item j
                if isInBubble j
                then
                    let Wjj' = Math.Pow(1.0 - float(dist j) / bubbleSize, learning.learningRate)
                    neuron.weights
                    |> Array.mapi (fun i w -> w + Wjj'*(input.[i] - w))
                    |> Neuron.create
                else neuron
            )

        // Calculate mean change
        let outsideBubble = float(learning.network.neuronsCount - bubbleSize)
        let meanChange =
            [|0..1|]
            |> Array.map (fun i ->
                attractedLayer.neurons
                |> Seq.mapi (fun j n -> (n.item i) - (layer.item j).item i)
                |> Seq.sum
               )
            |> Array.map (fun w -> w / outsideBubble)

        // Dispersion phase (restore mean position)
        let dispersedLayer =
            Layer.create layer.neuronsCount (fun j ->
                let neuron = attractedLayer.item j
                if isInBubble j
                then neuron
                else
                    neuron.weights
                    |> Array.mapi (fun i w -> w - meanChange.[i])
                    |> Neuron.create
            )

        { learning with network = { layer with neurons = dispersedLayer.neurons } }

module Geometry =
    let meanPoint (points: float[] seq) =
        [|
            points |> Seq.map (fun x->x.[0]) |> Seq.average
            points |> Seq.map (fun x->x.[1]) |> Seq.average
        |]

    let maxDiam (points: float[][]) =
        let x1 = points |> Seq.map (fun x->x.[0]) |> Seq.min
        let x2 = points |> Seq.map (fun x->x.[0]) |> Seq.max
        let y1 = points |> Seq.map (fun x->x.[1]) |> Seq.min
        let y2 = points |> Seq.map (fun x->x.[1]) |> Seq.max
        min (x2-x1) (y2-y1) // Max size of min bounding rectangle

    // Check the intersection of two segments (p1,p2) and (p3,p4)
    let isIntersect (p1,p2) (p3,p4) =
        let line (p1:float[]) (p2:float[]) =
            let A = p1.[1] - p2.[1]
            let B = p2.[0] - p1.[0]
            let C = p1.[0]*p2.[1] - p2.[0]*p1.[1]
            fun (p:float[]) ->
                let v = A*p.[0] + B*p.[1] + C
                if abs(v)<1e-6 then 0
                elif v>0.0 then 1 else -1
        let line1 = line p1 p2
        let line2 = line p3 p4

        (line1 p3 * line1 p4 < 0) &&
        (line2 p1 * line2 p2 < 0)

    // Update Layer to remove kinks (crosses of path segments)
    let removeKinks (layer:Layer) =
        for j1 = 0 to layer.neuronsCount - 1 do
            let p1 = layer.neurons.[j1]
            let p2 = layer.neurons.[(j1+1)%layer.neuronsCount]
            for j2 = 0 to j1 - 2 do
                let p3 = layer.neurons.[j2]
                let p4 = layer.neurons.[j2+1]
                if isIntersect
                    (p1.weights,p2.weights)
                    (p3.weights,p4.weights)
                then // reverse reading of [j2+1..j1] interval remove the kink
                    for i = j2+1 to (j1+(j2+1)+1)/2-1 do
                        let k = j1 - (i-j2-1)
                        let temp = layer.neurons.[i]
                        layer.neurons.[i] <- layer.neurons.[k]
                        layer.neurons.[k] <- temp
        layer

    // Square distance between two points
    let dist (p1:float[]) (p2:float[]) =
        (p1.[0]-p2.[0])*(p1.[0]-p2.[0]) + (p1.[1]-p2.[1])*(p1.[1]-p2.[1])

    // Blend two Layers(paths) to minimize length of result path
    let blend (layer1:Layer) (layer2:Layer) =
        // Here we are looking for pair of point
        // A in layer1 and B in layer2
        // in order to remove segments (A->D) and (C->B)
        // and add segment (A->B) and (C->D)
        // where D = A + 1 and C = B - 1
        let (_, (A, B)) =
            seq {
                for i=0 to layer1.neuronsCount-1 do
                    let i' = (layer1.neuronsCount + i + 1) % layer1.neuronsCount
                    let distAD = dist (layer1.neurons.[i].weights) (layer1.neurons.[i'].weights)
                    for j=0 to layer2.neuronsCount-1 do
                        let j' = (layer2.neuronsCount + j - 1) % layer2.neuronsCount
                        let distBC = dist (layer2.neurons.[j].weights) (layer2.neurons.[j'].weights)
                        let distAB = dist (layer1.neurons.[i].weights) (layer2.neurons.[j].weights)
                        let distDC = dist (layer1.neurons.[i'].weights) (layer2.neurons.[j'].weights)
                        // Here we tries to minimize the length of the blended path
                        // We remove arcs AD and BC and add arcs AB and DC
                        yield (distAB+distDC-distAD-distBC),(i,j)
            }
            |> Seq.minBy fst
        //let C = (layer2.neuronsCount + B - 1) % layer2.neuronsCount
        let D = (layer1.neuronsCount + A + 1) % layer1.neuronsCount
        Layer.create
            (layer1.neuronsCount + layer2.neuronsCount)
            (fun i ->
                if i < layer2.neuronsCount
                then
                    let ind = (B + i) % layer2.neuronsCount
                    layer2.neurons.[ind]
                else
                    let ind = (D + i - layer2.neuronsCount) % layer1.neuronsCount
                    layer1.neurons.[ind])
        |> removeKinks // Remove kinds after blend


module TravelingSantaProblem =
    open Geometry

    type ISantaSolver =
        // Execute: NumberOfIteraction -> uiUpdateFunc -> Final NN state
        abstract member Execute : int -> (int -> int -> Network -> Async<unit> ) -> Async<Network>

    type ElasticSolver(clusterId:int, network:Network, learningRate:float, cities:float[][]) =
        static member NetworkFromCities cities neurons =
            Network.createRandom 2 neurons
        static member ctor learningRate (id, network, cities) =
            ElasticSolver(id, network, learningRate, cities) :> ISantaSolver

        interface ISantaSolver with
          override __.Execute iterations updateUI =
            async {
                let trainer = NetworkLearning.create network
                let fixedLearningRate = learningRate / 20.
                let driftingLearningRate = fixedLearningRate * 19.
                let iterations = float iterations

                for i = 0 to (int iterations - 1) do

                    let learningRateUpdated = driftingLearningRate * (iterations - float i) / iterations + fixedLearningRate
                    let trainer = NetworkLearning.setLearningRate learningRateUpdated trainer
                    let learningRadiusUpdated = trainer.learningRadius * (iterations - float i) / iterations
                    let trainer = NetworkLearning.setLearningRadius learningRadiusUpdated trainer

                    let input = cities.[i % cities.Length]
                    let trainer = NetworkLearning.compute trainer input

                    if i % 1000 = 0 then
                        do! updateUI clusterId (i-1) trainer.network

                do! updateUI clusterId (int iterations - 1) trainer.network

                return trainer.network
            }

    // Kohonen network incorporating explicit statistics
    type KniesSolver(clusterId:int, network:Network, learningRate:float, cities:float[][]) =
        static member NetworkFromCities cities neurons =
            let c = meanPoint cities            // Center of bubble
            let r = (maxDiam cities) / 4.0      // Radius of bubble
            Network.createBubble neurons c r
        static member ctor learningRate (id, network, cities) =
            KniesSolver(id, network, learningRate, cities) :> ISantaSolver

        interface ISantaSolver with
          override __.Execute iterations updateUI =
            async {
                let mutable trainer = KNIES.create network

                for i = 0 to iterations - 1 do
                    let learningBubbleUpdated = 1 + (iterations - i) * (cities.Length /2) / iterations
                    trainer <- KNIES.setLearningBubble learningBubbleUpdated trainer
                    let Bt =
                        let BT = 1.0 + learningRate * float(cities.Length)
                        float(i) * BT / (float iterations)
                    trainer <- KNIES.setLearningRate Bt trainer

                    let input = cities.[i % cities.Length]
                    trainer <- KNIES.compute trainer input

                    if i % 1000 = 0 then
                        do! updateUI clusterId (i-1) trainer.network

                trainer <- { trainer with network = removeKinks trainer.network }
                do! updateUI clusterId (iterations - 1) trainer.network

                return trainer.network
            }


        // RUN your first function to found the best path using NN
        // run











    type ParallelSolver(neurons:int, cities:float[][], clusters:int,
                        networkCtor: float[][]->int->Network,
                        solverCtor:int*Network*float[][]->ISantaSolver) =

        let KMeanMaxIterations = 1000
        let kmeans clusters = // Run K-means clustering for cities
            let nearestCentroid centroids u =
                Array.minBy (dist u) centroids
            let updateCentroids (centroids) =
                cities
                |> Seq.groupBy (nearestCentroid centroids)
                |> Seq.map (snd >> meanPoint)
                |> Seq.sort
                |> Seq.toArray
            let rec update n centoids =
                let newCentroids = updateCentroids centoids
                let error =
                    if centoids.Length <> newCentroids.Length
                    then System.Double.MaxValue
                    else Array.fold2 (fun x u v -> x + dist u v) 0.0
                                centoids newCentroids
                if n=0 || (error < 1e-9)
                then newCentroids
                else update (n-1) newCentroids

            let centroids =
                Array.init clusters (fun _ -> cities.[rand.Next(cities.Length)])
                |> update KMeanMaxIterations
            cities |> Array.groupBy (nearestCentroid centroids)

        interface ISantaSolver with
            override __.Execute iterations updateUI =
              async {
                // clusterSize.[i] - size of cluster `i`
                // points.[i] - coordinates of centroid of cluster `i`
                // nns.[i] - async computation that
                let clusterSize, points, nns =
                    kmeans clusters // Create municipalities using k-means clustering
                    |> Array.mapi (fun i (c, mun) ->

                        // Divide neurons & iterations proportionally to cluster size
                        let munNeurons = neurons * mun.Length / cities.Length
                        let munIteration = iterations * mun.Length / cities.Length

                        let nn =
                            solverCtor(i, networkCtor mun munNeurons, mun)
                                .Execute munIteration updateUI
                        1, c, nn // `1` cluster with center in `c` and best path in `nn`
                       )
                    |> Array.unzip3

                let clusters = clusterSize.Length  // k-means may ends with less number of clusters than requested
                let canBlend =
                    Array.init clusters (fun a ->
                        let distance = // distance.[i] is distable between points `a` and `i`
                            Array.init clusters (fun i -> dist points.[a] points.[i])
                        let average = (Seq.sum distance) / float(clusters-1)
                        Array.init clusters (fun b ->
                            if b=a || distance.[b] > average+1.0
                            then false
                            else
                                seq {
                                    for i=0 to clusters-1 do
                                      for j=i+1 to clusters-1 do
                                        yield isIntersect (points.[a], points.[b])
                                                          (points.[i], points.[j])
                                }
                                |> Seq.exists id
                                |> not
                        )
                    )

                // parent.[i] is ID of parent cluster after blend, or `-1` if cluster is not blended
                let parent = Array.create clusters -1
                let rec getClustedId i =
                    if parent.[i] = -1 then i
                    else getClustedId(parent.[i])

                for __=1 to clusters-1 do                               // N-1 merged to build 1 cluster
                    let A =                                             // first merge candidate
                        [|0..clusters-1|]
                        |> Array.filter (fun x -> parent.[x] = -1)      // Not merged
                        |> Array.minBy (fun x -> clusterSize.[x])       // Choose Min cluster by size
                    let inA =                                           // Clusters merged with cluster `A`
                        [|0..clusters-1|]
                        |> Array.filter (fun x -> (getClustedId x) = A)
                    let B =                                             // second merge candidate
                        let mergeCandidate =
                            [|0..clusters-1|]
                            |> Array.filter (fun x ->                   // Choose those that could be merged with A
                                inA |> Seq.exists (fun y -> canBlend.[y].[x]))
                        if mergeCandidate.Length <> 0
                        then
                            mergeCandidate
                            |> Array.map (getClustedId)                 // Map to parent cluster if already merged
                            |> Array.distinct                           // Remove duplicated
                            |> Array.filter ((<>)A)                     // Exclude itself (if any)
                            |> Array.minBy (fun x -> clusterSize.[x])   // Choose Min cluster by size
                        else
                            // If there is no merge candidates according to `canBlend` matrix
                            // choose the closest cluster that is not blended yet
                            [|0..clusters-1|]
                            |> Array.filter (fun x -> parent.[x] = -1 && x<>A)
                            |> Array.minBy (fun x -> dist points.[A] points.[x])

                    // Merge `A` and `B`
                    parent.[B] <- A
                    clusterSize.[A] <- clusterSize.[A] + clusterSize.[B]
                    clusterSize.[B] <- 0
                    let nnA, nnB = nns.[A], nns.[B]
                    nns.[A] <- async {
                        let! nn = Async.Parallel [nnA;nnB]              // Calculate sub-paths in parallel
                        let network = blend nn.[0] nn.[1] |> removeKinks// Blend paths
                        do! updateUI B iterations Network.zero          // Hide blended cluster
                        do! updateUI A iterations network               // Update new master-cluster
                        //do! Async.Sleep 1000                          // Uncomment to visualize merge
                        return network
                    }

                let root =
                    [|0..clusters-1|]  // Pick the last/root cluster in the merge tree
                    |> Seq.pick (fun i -> if parent.[i] = -1 then Some(i) else None)
                return! nns.[root]
                // We cannot run NN here one more time, because initial learning parameters are very aggressive
                // But we can improve algorithm to be able start it in gently mode
              }



open TravelingSantaProblem

type MainViewModel() as this =
    inherit ViewModelBase()


    let mutable cts = new CancellationTokenSource()

    let pointsStream = Event<(float * float)[]>()
    let livePointsChart =
        pointsStream.Publish
        |> Observable.map id
        |> LiveChart.Point


    let createChart pathes =
        let pathStreams = List.init pathes (fun _ -> Event<(float * float)[]>())
        let pathObs = pathStreams |> List.map (fun s -> s.Publish |> Observable.map(id))

        let livePathCharts = pathObs |> List.map (LiveChart.Line)
        let chartCombine = Chart.Combine(livePointsChart :: livePathCharts).WithYAxis(Enabled=false).WithXAxis(Enabled=false)

        let chart = new ChartControl(chartCombine)
        let host = new WindowsFormsHost(Child = chart)

        pathStreams, host


    let cities = this.Factory.Backing(<@ this.Cities @>, 100)
    let iterations = this.Factory.Backing(<@ this.Iterations @>, 25000)

    // To avoid oscillation of neurons between different cities, they proposed that
    // the number of neurons should be greater than number of cities (M >= 3N).
    // In our study we assume fixed number of neurons (M = 5N)
    let neurons = this.Factory.Backing(<@ this.Neurons @>, 4*100)
    let learningRate = this.Factory.Backing(<@ this.LearningRate @>, 0.005)
    let clusters = this.Factory.Backing(<@ this.Clusters @>, 4)

    let currentIterations = this.Factory.Backing(<@ this.CurrentIterations @>, 0)
    let executionTime = this.Factory.Backing(<@ this.ExecutionTime @>, "")
    do mapAgent.PostAndReply(fun ch -> SetMap(cities.Value, ch)) |> pointsStream.Trigger

    let mutable pathStreams, host = createChart 1
    let hostChart = this.Factory.Backing(<@ this.Chart @>, host)

    let initControls n =
        this.CurrentIterations <- 0
        this.ExecutionTime <- ""
        pointsStream.Trigger [||]
        mapAgent.PostAndReply(fun ch -> SetMap(n, ch)) |> pointsStream.Trigger
        pathStreams |> Seq.iter (fun stream -> stream.Trigger [||])

    let onCancel _ =
        this.CurrentIterations <- 0
        this.ExecutionTime <- ""
        pathStreams |> Seq.iter (fun stream -> stream.Trigger [||])
        cts.Dispose()
        cts <- new CancellationTokenSource()
        this.StartElasticCommand.CancellationToken <- cts.Token
        this.StartKniesCommand.CancellationToken <- cts.Token
        this.StartElasticInParallelCommand.CancellationToken <- cts.Token
        this.StartKniesInParallelCommand.CancellationToken <- cts.Token

    let cancelClear () =
        cts.Cancel()

    let cancel =
        this.Factory.CommandSyncChecked(cancelClear, (fun _ -> this.OperationExecuting), [ <@@ this.OperationExecuting @@> ])

    let initPoints =
        this.Factory.CommandSyncParamChecked(initControls, (fun _ -> not this.OperationExecuting), [ <@@ this.OperationExecuting @@> ])

    let createCommand getPathCount createSolver =
        this.Factory.CommandAsync((fun ui -> async {
            let streams, host = createChart (getPathCount())
            pathStreams <- streams
            hostChart.Value <- host

            let updateControl streamId currentIteration (network:Network) =
                async {
                    let path =
                      if network = Network.zero then [||]
                      else
                        Array.init (network.neuronsCount + 1) (fun id ->
                            let n = id % network.neuronsCount
                            (network.item n).item 0, (network.item n).item 1)

                    do! Async.SwitchToContext ui
                    this.CurrentIterations <- (currentIteration + 1)
                    streams.[streamId].Trigger path
                    do! Async.SwitchToThreadPool()
                }

            let time = System.Diagnostics.Stopwatch.StartNew()
            let! cities = mapAgent.PostAndAsyncReply(fun ch -> GetMap(cities.Value, ch))
            let cities = cities |> Array.map (fun (x,y) -> [|x;y|])

            let tsp = createSolver(cities) :> ISantaSolver
            let! _ = tsp.Execute iterations.Value updateControl

            this.ExecutionTime <- sprintf "Time %d ms" time.ElapsedMilliseconds
        }), token=cts.Token, onCancel=onCancel)

    let startElastic =
        createCommand
            (fun() -> 1)
            (fun cities ->
                let network = ElasticSolver.NetworkFromCities cities neurons.Value
                ElasticSolver(0, network, learningRate.Value, cities) )

    let startKnies =
        createCommand
            (fun() -> 1)
            (fun cities ->
                let network = KniesSolver.NetworkFromCities cities neurons.Value
                KniesSolver(0, network, learningRate.Value, cities) )

    let startElasticInParallel =
        createCommand
            (fun() -> clusters.Value)
            (fun cities ->
                ParallelSolver(neurons.Value, cities, clusters.Value,
                               ElasticSolver.NetworkFromCities,
                               ElasticSolver.ctor learningRate.Value)
            )

    let startKniesInParallel =
        createCommand
            (fun() -> clusters.Value)
            (fun cities ->
                ParallelSolver(neurons.Value, cities, clusters.Value,
                               KniesSolver.NetworkFromCities,
                               KniesSolver.ctor learningRate.Value)
            )

    do initControls (cities.Value)

    member this.Chart
        with get () = hostChart.Value
        and set value = hostChart.Value <- value

    member this.Cities
        with get () = cities.Value
        and set value = cities.Value <- value

    member this.Neurons
        with get () = neurons.Value
        and set value = neurons.Value <- value

    member this.LearningRate
        with get () = learningRate.Value
        and set value = learningRate.Value <- value

    member this.Iterations
        with get () = iterations.Value
        and set value = iterations.Value <- value

    member this.Clusters
        with get () = clusters.Value
        and set value = clusters.Value <- value

    member this.CurrentIterations
        with get () = currentIterations.Value
        and set value = currentIterations.Value <- value

    member this.ExecutionTime
        with get () = executionTime.Value
        and set value = executionTime.Value <- value

    member this.InitPointsCommand = initPoints
    member this.StartElasticCommand : IAsyncNotifyCommand = startElastic
    member this.StartKniesCommand : IAsyncNotifyCommand = startKnies
    member this.StartElasticInParallelCommand : IAsyncNotifyCommand = startElasticInParallel
    member this.StartKniesInParallelCommand : IAsyncNotifyCommand = startKniesInParallel
    member this.CancelCommand = cancel


