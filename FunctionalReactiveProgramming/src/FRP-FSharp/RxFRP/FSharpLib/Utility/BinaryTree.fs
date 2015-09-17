namespace Easj360FSharp

module Structers =

    open System

    type BinaryTree<'a> =
        | Node of 'a * BinaryTree<'a> * BinaryTree<'a>
        | Data of 'a
        | Empty
        member bt.iter(fn: ('a) -> unit) =
            match bt with
            | Empty     -> ()
            | Node(data, left, right) ->
                left.iter(fn)
                fn(data)
                right.iter(fn)
            | Data(data)    -> fn(data)

    type LazyBinaryTree<'a> =
        | Node of Lazy<'a * LazyBinaryTree<'a> * LazyBinaryTree<'a>>
        | Empty
        member bt.iter(fn: ('a) -> unit) =
            match bt with
            | Empty         -> ()
            | Node(item)    ->  let data, left, right = item.Force()                                
                                left.iter(fn)
                                fn(data)
                                right.iter(fn)

     //let bt = Node(lazy("Ricky", Empty, Empty)) 

     //let bt3 = Node(lazy("Ricky", Node(lazy("Bugghina", Node(lazy("Bryony", Node(lazy("Aldo",Empty, Empty)), Empty)), Empty)),Empty))
