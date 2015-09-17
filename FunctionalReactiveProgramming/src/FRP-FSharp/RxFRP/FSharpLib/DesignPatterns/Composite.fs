module Composite

// define visitor interface
type IVisitor<'T> =
    abstract member Do : 'T -> unit

// define a composite node
type CompositeNode<'T> = 
    | Node of 'T
    | Tree of 'T * CompositeNode<'T> * CompositeNode<'T>
    with 
        // define in-order traverse
        member this.InOrder f = 
            match this with
            | Tree(n, left, right) -> 
                left.InOrder f
                f n
                right.InOrder(f)
            | Node(n) -> f n
        
        // define pre-order traverse
        member this.PreOrder f =
            match this with
            | Tree(n, left, right) ->                 
                f n
                left.PreOrder f
                right.PreOrder f
            | Node(n) -> f n

        // define post order traverse
        member this.PostOrder f =
            match this with
            | Tree(n, left, right) -> 
                left.PostOrder f
                right.PostOrder f
                f n
            | Node(n) -> f n

let invoke() = 
    // define a tree structure
    let tree = Tree(1, Tree(11, Node(12), Node(13)), Node(2))

    // define a visitor, it gets the summary of the node values
    let wrapper = 
        let result = ref 0
        ({ new IVisitor<int> with                
                member this.Do n = 
                    result := !result + n                
        }, result)

    // pre-order iterates the tree and prints out the result
    tree.PreOrder (fst wrapper).Do
    printfn "result = %d" !(snd wrapper)
