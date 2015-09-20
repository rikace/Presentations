namespace Easj360FSharp

open System

//type HuffmanCoder(symbols : seq<char>, frequencies : seq<float>) =  
        /// Huffman coding uses a binary tree whose leaves are the input symbols 
        /// and whose internal nodes are the combined expected frequency of all the
        /// symbols beneath them.
        type HuffmanTree = 
            | Leaf of char * float
            | Node of float * HuffmanTree * HuffmanTree

        /// Provides encoding and decoding for strings containing the given symbols and expected frequencies
        type HuffmanCoder(symbols: seq<char>, frequencies : seq<float>) =
           
            /// Builds a list of leafs for a huffman encoding tree from the input frequencies
            let huffmanTreeLeafs =    
                Seq.zip symbols frequencies
                |> Seq.toList
                |> List.map Leaf
                
            /// Utility function to get the frequency from a huffman encoding tree node
            let frequency node =
                match node with
                | Leaf(_,p) -> p
                | Node(p,_,_) -> p    

            /// Builds a huffman encoding tree from a list of root nodes, iterating until a unique root node
            let rec buildCodeTree roots = 
                match roots |> List.sortBy frequency with
                | [] -> failwith "Cannot build a Huffman Tree for no inputs" 
                | [node] -> node
                | least::nextLeast::rest -> 
                           let combinedFrequency = frequency least + frequency nextLeast
                           let newNode = Node(combinedFrequency, least, nextLeast)
                           buildCodeTree (newNode::rest)
                       
            let tree = buildCodeTree huffmanTreeLeafs
             
            /// Builds a table of huffman codes for all the leafs in a huffman encoding tree
            let huffmanCodeTable = 
                let rec huffmanCodes tree = 
                    match tree with
                    | Leaf (c,_) -> [(c, [])]
                    | Node (_, left, right) -> 
                        let leftCodes = huffmanCodes left |> List.map (fun (c, code) -> (c, true::code))
                        let rightCodes = huffmanCodes right |> List.map (fun (c, code) -> (c, false::code))
                        List.append leftCodes rightCodes
                huffmanCodes tree 
                |> List.map (fun (c,code) -> (c,List.toArray code))
                |> Map.ofList

            /// Encodes a string using the huffman encoding table
            let encode (str:string) = 
                let encodeChar c = 
                    match huffmanCodeTable |> Map.tryFind c with
                    | Some bits -> bits
                    | None -> failwith "No frequency information provided for character '%A'" c
                str.ToCharArray()
                |> Array.map encodeChar
                |> Array.concat
               
            
            /// Decodes an array of bits into a string using the huffman encoding tree
            let decode bits =
                let rec decodeInner bitsLeft treeNode result = 
                    match bitsLeft, treeNode with
                    | [] , Node (_,_,_) -> failwith "Bits provided did not form a complete word"
                    | [] , Leaf (c,_) ->  (c:: result) |> List.rev |> List.toArray
                    | _  , Leaf (c,_) -> decodeInner bitsLeft tree (c::result)
                    | b::rest , Node (_,l,r)  -> if b
                                                 then decodeInner rest l result
                                                 else decodeInner rest r result
                let bitsList = Array.toList bits
                new String(decodeInner bitsList tree [])
                         
            member coder.Encode source = encode source
            member coder.Decode source = decode source
            
    
    (*Pattern Matching 
Pattern matching is one of the basic but very powerful features of F#.  Using pattern matching allows code to be succinct while at the same time very clear about it's behaviour.  Just about every function in the code above uses pattern matching - which is typical of a lot of F# code.  

In simple cases, such as in "huffmanCodes" above, pattern matching makes it easy to switch on the possible cases of a union datastructure.  

match tree with
| Leaf (c,_) -> //...
| Node (_, left, right) -> //...In more complex cases like "decodeInner" above, pattern matching helps to guide your code. You list each shape of input that you know how to handle, and the leaf nodes of the pattern you matched indicate the data needed to define the behaviour of that case.  Then the compiler kindly tells you which cases you haven't covered. When I wrote this function originally, I was missing the first case, and got this helpful compiler warning:

Warning: Incomplete pattern matches on this expression. The value '([],Node (_, _, _))' will not be matched Which was exactly right - this particular input indicates that the user provided illegal inputs!

 

Pipelining
Pipelining is a nice way to declaratively describe a series of operations to perform on an input.  It's philosophically just like pipelining in a command shell - take the input data from the left and pass it as input to the function on the right.  

Because the F# libraries provide a good collection of primitive operations to perform on your data, it's easy to describe a transformation as a pipeline of data through a sequence of these operations.  For example, you can filter, project, collapse, zip, and re-package data easily and declaratively.

 

Collections
The code uses the 4 common F#/.NET collections:

F# "List": Immutable linked list used for recursive algorithms which walk across a list 
F# "Map": Immutable dictionary used to store the codes for each symbols 
F# "Seq" = .NET "IEnumerable":  Primitive collection interface, used for the inputs 
.NET "Array": Basic typed arrays used as the output of the encoding 
Note that it is easy and common to switch between these collections as needed, using functions like "List.to_array" and "Map.of_list".

 

"It's a .NET component!" a.k.a. "The Slickness"
When I wrote this code, I started out in "experimentation" mode. I just wrote a bunch of small functions at the top level and used F# Interactive to execute them as I went. But once I had something that seemed to work, I wanted to wrap it up in a class that could encapsulate all the functionality, and provide a nice .NET interface. Here's what I did:

Tab everything in one level
Wrap the code in:


type HuffmanCoder(symbols : seq<char>, frequencies : seq<float>) =         // All the code goes here...        member coder.Encode source = encode source    member coder.Decode source = decode sourceI can't really understate how amazing this is.  In F#, it is super-simple to smoothly transition from experimentation coding to component design coding.  This is something that F# can manage because it is a hybrid functional/OO language intergated carefully into .NET.  And because of this, it is really easy to build components of a larger .NET system in F#.  I heard this referred to once as "the slickness" of functional OO coding in F#, and I like that term, because whenever I do this it just feels so slick. :-)

In case you are curious, here's what it looks like from C#:

    public class HuffmanCoder
    {
        public HuffmanCoder(IEnumerable<char> symbols, IEnumerable<double> frequencies);

        public string Decode(bool[] source);
        public bool[] Encode(string source);
    }*)