// [snippet: Ninety-Nine F# Problems - Problems 1 - 10 - Lists]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
// [/snippet]

// [snippet: (*) Problem 1 : Find the last element of a list.]
/// Example in F#: 
/// > myLast [1; 2; 3; 4];;
/// val it : int = 4
/// > myLast ['x';'y';'z'];;
/// val it : char = 'z'

(*[omit:(Solution 1)]*)
// Solution using recursion
let rec myLast xs = 
    match xs with
        | [] -> failwith "empty list you fool!"
        | [x] -> x
        | _::xs -> myLast xs
(*[/omit]*)

(*[omit:(Solution 2)]*)
// Solution using higher-order functions
let myLast' xs = xs |> List.rev |> List.head
(*[/omit]*)

(*[omit:(Solution 3)]*)
// ignore the acumulator using reduce
let myLast'' xs = List.reduce(fun _ x -> x) xs
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 2 : Find the last but one element of a list.]
/// (Note that the Lisp transcription of this problem is incorrect.) 
///
/// Example in F#: 
/// myButLast [1; 2; 3; 4];;
/// val it : int = 3
/// > myButLast ['a'..'z'];;
/// val it : char = 'y'

(*[omit:(Solution 1)]*)
// Solution with pattern matching
let rec myButLast = function
    | [] -> failwith "empty list you fool!"
    | [x] -> failwith "singleton list you fool!"
    | [x;_] -> x
    | _::xs -> myButLast xs
(*[/omit]*)

(*[omit:(Solution 2)]*)
let myButLast' xs = xs |> List.rev |> List.tail |> List.head
(*[/omit]*)

(*[omit:(Solution 3)]*)
let myButLast'' xs = 
    let flip f a b = f b a
    xs |> List.rev |> flip List.nth 1
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 3 : Find the K'th element of a list. The first element in the list is number 1.]
/// Example: 
/// * (element-at '(a b c d e) 3)
/// c
/// 
/// Example in F#: 
/// > elementAt [1; 2; 3] 2;;
/// val it : int = 2
/// > elementAt (List.ofSeq "fsharp") 5;;
/// val it : char = 'r'

(*[omit:(Solution 1)]*)
// List.nth is zero based
let elementAt xs n = List.nth xs (n - 1)
(*[/omit]*)

(*[omit:(Solution 2)]*)
// Recursive solution with pattern matching
let rec elementAt' xs n = 
    match xs,n with
        | [],_   -> failwith "empty list you fool!"
        | x::_,1 -> x
        | _::xs,n -> elementAt xs (n - 1)
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 4 : Find the number of elements of a list.]
/// Example in F#: 
/// 
/// > myLength [123; 456; 789];;
/// val it : int = 3
/// > myLength <| List.ofSeq "Hello, world!"
/// val it : int = 13 

(*[omit:(Solution 1)]*)
// Solution using the library method
let myLength = List.length
(*[/omit]*)

(*[omit:(Solution 2)]*)
// replace the elemt with 1 and sum all the ones
let myLength' xs = xs |> List.sumBy(fun _ -> 1) 
(*[/omit]*)

(*[omit:(Solution 3)]*)
// Solution using tail-recursion
let myLength'' xs =
    let rec length acc = function
        | [] -> acc
        | _::xs  -> length (acc+1) xs
    length 0 xs
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 5 : Reverse a list.]
/// Example in F#: 
///
/// > reverse <| List.ofSeq ("A man, a plan, a canal, panama!")
/// val it : char list =
///  ['!'; 'a'; 'm'; 'a'; 'n'; 'a'; 'p'; ' '; ','; 'l'; 'a'; 'n'; 'a'; 'c'; ' ';
///   'a'; ' '; ','; 'n'; 'a'; 'l'; 'p'; ' '; 'a'; ' '; ','; 'n'; 'a'; 'm'; ' ';
///   'A']
/// > reverse [1,2,3,4];;
/// val it : int list = [4; 3; 2; 1]

(*[omit:(Solution 1)]*)
// Using tail-recursion
let reverse xs = 
    let rec rev acc = function
        | [] -> acc
        | x::xs -> rev (x::acc) xs
    rev [] xs
(*[/omit]*)

(*[omit:(Solution 2)]*)
let reverse' xs = List.fold(fun acc x -> x::acc) [] xs
(*[/omit]*)

(*[omit:(Solution 3)]*)
let reverse'' = List.rev
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 6 : Find out whether a list is a palindrome.]
/// A palindrome can be read forward or backward; e.g. (x a m a x).
/// 
/// Example in F#: 
/// > isPalindrome [1;2;3];;
/// val it : bool = false
/// > isPalindrome <| List.ofSeq "madamimadam";;
/// val it : bool = true
/// > isPalindrome [1;2;4;8;16;8;4;2;1];;
/// val it : bool = true

(*[omit:(Solution)]*)
// A list is a palindrome is the list is equal to its reverse
let isPalindrome xs = xs = List.rev xs
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 7 : Flatten a nested list structure.]
/// Transform a list, possibly holding lists as elements into a `flat' list by replacing each 
/// list with its elements (recursively).
///  
/// Example: 
/// * (my-flatten '(a (b (c d) e)))
/// (A B C D E)
///  
/// Example in F#: 
/// 
type 'a NestedList = List of 'a NestedList list | Elem of 'a
///
/// > flatten (Elem 5);;
/// val it : int list = [5]
/// > flatten (List [Elem 1; List [Elem 2; List [Elem 3; Elem 4]; Elem 5]]);;
/// val it : int list = [1;2;3;4;5]
/// > flatten (List [] : int NestedList);;
/// val it : int list = []

(*[omit:(Solution 1)]*)
let flatten ls = 
    let rec loop acc = function 
        | Elem x -> x::acc
        | List xs -> List.foldBack(fun x acc -> loop acc x) xs acc
    loop [] ls
(*[/omit]*)

(*[omit:(Solution 2)]*)
#nowarn "40"
let flatten' x =
    let rec loop = List.collect(function
        | Elem x -> [x]
        | List xs -> loop xs)
    loop [x]
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 8 : Eliminate consecutive duplicates of list elements.] 
/// If a list contains repeated elements they should be replaced with a single copy of the 
/// element. The order of the elements should not be changed.
///  
/// Example: 
/// * (compress '(a a a a b c c a a d e e e e))
/// (A B C A D E)
///  
/// Example in F#: 
/// 
/// > compress ["a";"a";"a";"a";"b";"c";"c";"a";"a";"d";"e";"e";"e";"e"];;
/// val it : string list = ["a";"b";"c";"a";"d";"e"]

(*[omit:(Solution 1)]*)
let compress xs = List.foldBack(fun x acc -> if List.isEmpty acc then [x] elif x = List.head acc then acc else x::acc) xs []
(*[/omit]*)

(*[omit:(Solution 2)]*)
let compress' = function
    | [] -> []
    | x::xs -> List.fold(fun acc x -> if x = List.head acc then acc else x::acc) [x] xs |> List.rev
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 9 : Pack consecutive duplicates of list elements into sublists.] 
/// If a list contains repeated elements they should be placed 
/// in separate sublists.
///  
/// Example: 
/// * (pack '(a a a a b c c a a d e e e e))
/// ((A A A A) (B) (C C) (A A) (D) (E E E E))
///  
/// Example in F#: 
/// 
/// > pack ['a'; 'a'; 'a'; 'a'; 'b'; 'c'; 'c'; 'a'; 
///         'a'; 'd'; 'e'; 'e'; 'e'; 'e']
/// val it : char list list =
///  [['a'; 'a'; 'a'; 'a']; ['b']; ['c'; 'c']; ['a'; 'a']; ['d'];
///   ['e'; 'e'; 'e'; 'e']]

(*[omit:(Solution)]*)
let pack xs = 
    let collect x = function
        | (y::xs)::xss when x = y -> (x::y::xs)::xss
        | xss -> [x]::xss
    List.foldBack collect xs []
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 10 : Run-length encoding of a list.]
/// Use the result of problem P09 to implement the so-called run-length 
/// encoding data compression method. Consecutive duplicates of elements 
/// are encoded as lists (N E) where N is the number of duplicates of the element E.
///  
/// Example: 
/// * (encode '(a a a a b c c a a d e e e e))
/// ((4 A) (1 B) (2 C) (2 A) (1 D)(4 E))
///  
/// Example in F#: 
/// 
/// encode <| List.ofSeq "aaaabccaadeeee"
/// val it : (int * char) list =
///   [(4,'a');(1,'b');(2,'c');(2,'a');(1,'d');(4,'e')]

(*[omit:(Solutions 1)]*)
let encode xs = xs |> pack |> List.map (Seq.countBy id >> Seq.head >> fun(a,b)-> b,a)
(*[/omit]*)

(*[omit:(Solutions 2)]*)
let encode' xs = xs |> pack |> List.map(fun xs -> List.length xs, List.head xs)
(*[/omit]*)

(*[omit:(Solutions 3)]*)
let encode'' xs = 
    let collect x = function
        | [] -> [(1, x)]
        | (n,y)::xs as acc-> 
            if x = y then
                (n+1, y)::xs
            else
                (1,x)::acc
    List.foldBack collect xs []
(*[/omit]*)
// [/snippet]// [snippet: Ninety-Nine F# Problems - Problems 11 - 20 - List, continued]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
// [/snippet]

// [snippet: (*) Problem 11 :  Modified run-length encoding.]
/// Modify the result of problem 10 in such a way that if an element has no duplicates it 
/// is simply copied into the result list. Only elements with duplicates are transferred as
/// (N E) lists.
///  
/// Example: 
/// * (encode-modified '(a a a a b c c a a d e e e e))
/// ((4 A) B (2 C) (2 A) D (4 E))
///  
/// Example in F#: 
/// 
/// > encodeModified <| List.ofSeq "aaaabccaadeeee"
/// val it : char Encoding list =
///   [Multiple (4,'a'); Single 'b'; Multiple (2,'c'); Multiple (2,'a');
///    Single 'd'; Multiple (4,'e')]

type 'a Encoding = Multiple of int * 'a | Single of 'a

(*[omit:(Solution)]*)
/// From problem 9 
let pack xs = 
    let collect x = function
        | (y::xs)::xss when x = y -> (x::y::xs)::xss
        | xss -> [x]::xss
    List.foldBack collect xs []

let encodeModified xs = xs |> pack |> List.map (Seq.countBy id >> Seq.head >> fun(x,n)-> if n = 1 then Single x else Multiple (n,x))
(*[/omit]*)
// [/snippet]


// [snippet: (**) Problem 12 : Decode a run-length encoded list.]
/// Given a run-length code list generated as specified in problem 11. Construct its 
/// uncompressed version.
///  
/// Example in F#: 
/// 
/// > decodeModified 
///     [Multiple (4,'a');Single 'b';Multiple (2,'c');
///      Multiple (2,'a');Single 'd';Multiple (4,'e')];;
/// val it : char list =
///   ['a'; 'a'; 'a'; 'a'; 'b'; 'c'; 'c'; 'a'; 'a'; 'd'; 'e'; 'e'; 'e'; 'e']

(*[omit:(Solution)]*)
let decodeModified xs = 
    let expand = function
        | Single x -> [x]
        | Multiple (n,x) -> List.replicate n x
    xs |> List.collect expand
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 13 : Run-length encoding of a list (direct solution).]
/// Implement the so-called run-length encoding data compression method directly. I.e. 
/// don't explicitly create the sublists containing the duplicates, as in problem 9, 
/// but only count them. As in problem P11, simplify the result list by replacing the 
/// singleton lists (1 X) by X.
///  
/// Example: 
/// * (encode-direct '(a a a a b c c a a d e e e e))
/// ((4 A) B (2 C) (2 A) D (4 E))
///  
/// Example in F#: 
/// 
/// > encodeDirect <| List.ofSeq "aaaabccaadeeee"
/// val it : char Encoding list =
///   [Multiple (4,'a'); Single 'b'; Multiple (2,'c'); Multiple (2,'a');
///    Single 'd'; Multiple (4,'e')]

(*[omit:(Solution)]*)
let encodeDirect xs = 
    let collect x = function
        | [] -> [Single x]
        | Single y::xs when x = y -> Multiple(2, x)::xs
        | Single _::_ as xs -> Single x::xs
        | Multiple(n,y)::xs when y = x -> Multiple(n + 1, x)::xs
        | xs -> Single x::xs
    List.foldBack collect xs []
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 14 : Duplicate the elements of a list.]
/// Example: 
/// * (dupli '(a b c c d))
/// (A A B B C C C C D D)
///  
/// Example in F#: 
/// 
/// > dupli [1; 2; 3]
/// [1;1;2;2;3;3]

(*[omit:(Solution 1)]*)
let dupli xs = xs |> List.map (fun x -> [x; x]) |> List.concat
(*[/omit]*)

(*[omit:(Solution 2)]*)
let rec dupli' = function
    | [] -> []
    | x::xs -> x::x::dupli' xs
(*[/omit]*)

(*[omit:(Solution 3)]*)
let dupli'' xs = [ for x in xs do yield x; yield x ]
(*[/omit]*)

(*[omit:(Solution 4)]*)
let dupli''' xs = xs |> List.collect (fun x -> [x; x])
(*[/omit]*)

(*[omit:(Solution 5)]*)
let dupli'''' xs = (xs,[]) ||> List.foldBack(fun x xs -> x::x::xs) 
(*[/omit]*)

(*[omit:(Solution 6)]*)
let dupli''''' xs = ([], xs) ||> List.fold(fun xs x -> xs @ [x; x])
(*[/omit]*)

(*[omit:(Solution 7)]*)
let dupli'''''' xs = xs |> List.collect (List.replicate 2)
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 15 : Replicate the elements of a list a given number of times.]
/// Example: 
/// * (repli '(a b c) 3)
/// (A A A B B B C C C)
///  
/// Example in F#: 
/// 
/// > repli (List.ofSeq "abc") 3
/// val it : char list = ['a'; 'a'; 'a'; 'b'; 'b'; 'b'; 'c'; 'c'; 'c']

(*[omit:(Solution 1)]*)
let repli xs n = xs |> List.collect (List.replicate n)
(*[/omit]*)

(*[omit:(Solution 2)]*)
let repli' xs n= 
    [ for x in xs do 
        for i=1 to n do
            yield x ]
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 16 : Drop every N'th element from a list.]
/// Example: 
/// * (drop '(a b c d e f g h i k) 3)
/// (A B D E G H K)
///  
/// Example in F#: 
/// 
/// > dropEvery (List.ofSeq "abcdefghik") 3;;
/// val it : char list = ['a'; 'b'; 'd'; 'e'; 'g'; 'h'; 'k']

(*[omit:(Solution 1)]*)
let dropEvery xs n = xs |> List.mapi (fun i x -> (i + 1,x)) |> List.filter(fun (i,_) -> i % n <> 0) |> List.map snd
(*[/omit]*)

(*[omit:(Solution 2)]*)
let dropEvery' xs n =
    let rec drop xs count =
        match xs,count with
            | [], _ -> []
            | _::xs, 1 -> drop xs n
            | x::xs, _ -> x::drop xs (count - 1) 
    drop xs n
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 17 : Split a list into two parts; the length of the first part is given.]
/// Do not use any predefined predicates. 
/// 
/// Example: 
/// * (split '(a b c d e f g h i k) 3)
/// ( (A B C) (D E F G H I K))
///  
/// Example in F#: 
/// 
/// > split (List.ofSeq "abcdefghik") 3
/// val it : char list * char list =
///   (['a'; 'b'; 'c'], ['d'; 'e'; 'f'; 'g'; 'h'; 'i'; 'k'])

(*[omit:(Solution)]*)
let split xs n = 
    let rec take n xs =
        match xs,n with
            | _,0 -> []
            | [],_ -> []
            | x::xs,n -> x::take (n-1) xs
    let rec drop n xs =
        match xs,n with
            | xs,0 -> xs
            | [],_ -> []
            | _::xs,n -> drop (n-1) xs
    take n xs, drop n xs
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 18 : Extract a slice from a list.]
/// Given two indices, i and k, the slice is the list containing the elements between the 
/// i'th and k'th element of the original list (both limits included). Start counting the 
/// elements with 1.
///  
/// Example: 
/// * (slice '(a b c d e f g h i k) 3 7)
/// (C D E F G)
///  
/// Example in F#: 
/// 
/// > slice ['a';'b';'c';'d';'e';'f';'g';'h';'i';'k'] 3 7;;
/// val it : char list = ['c'; 'd'; 'e'; 'f'; 'g']

(*[omit:(Solution 1)]*)
let slice xs s e =
    let rec take n xs =
        match xs,n with
            | _,0 -> []
            | [],_ -> []
            | x::xs,n -> x::take (n-1) xs
    let rec drop n xs =
        match xs,n with
            | xs,0 -> xs
            | [],_ -> []
            | _::xs,n -> drop (n-1) xs
    let diff = e - s
    xs |> drop (s - 1) |> take (diff + 1)
(*[/omit]*)

(*[omit:(Solution 2)]*)
let slice' xs s e = [ for (x,j) in Seq.zip xs [1..e] do
                            if s <= j then
                                yield x ]
(*[/omit]*)

(*[omit:(Solution 3)]*)
let slice'' xs s e = xs |> Seq.zip (seq {1 .. e}) |> Seq.filter(fst >> (<=) s) |> Seq.map snd
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 19 : Rotate a list N places to the left.]
/// Hint: Use the predefined functions length and (@) 
/// 
/// Examples: 
/// * (rotate '(a b c d e f g h) 3)
/// (D E F G H A B C)
/// 
/// * (rotate '(a b c d e f g h) -2)
/// (G H A B C D E F)
///  
/// Examples in F#: 
/// 
/// > rotate ['a';'b';'c';'d';'e';'f';'g';'h'] 3;;
/// val it : char list = ['d'; 'e'; 'f'; 'g'; 'h'; 'a'; 'b'; 'c']
///  
/// > rotate ['a';'b';'c';'d';'e';'f';'g';'h'] (-2);;
/// val it : char list = ['g'; 'h'; 'a'; 'b'; 'c'; 'd'; 'e'; 'f']

(*[omit:(Solution 1)]*)
// using problem 17
let rotate xs n =
    let at = let ln = List.length xs in abs <| (ln + n) % ln
    let st,nd = split xs at
    nd @ st
(*[/omit]*)

(*[omit:(Solution 2)]*)
let rec rotate' xs n =
    match xs, n with
        | [], _ -> []
        | xs, 0 -> xs
        | x::xs, n when n > 0 -> rotate' (xs @ [x]) (n - 1)
        | xs, n -> rotate' xs (List.length xs + n)
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 20 : Remove the K'th element from a list.]
/// Example in Prolog: 
/// ?- remove_at(X,[a,b,c,d],2,R).
/// X = b
/// R = [a,c,d]
///  
/// Example in Lisp: 
/// * (remove-at '(a b c d) 2)
/// (A C D)
///  
/// (Note that this only returns the residue list, while the Prolog version also returns 
/// the deleted element.)
///  
/// Example in F#: 
/// 
/// > removeAt 1 <| List.ofSeq "abcd";;
/// val it : char * char list = ('b', ['a'; 'c'; 'd'])

(*[omit:(Solution 1)]*)
let removeAt n xs = 
    let rec rmAt acc xs n =
        match xs, n with
            | [], _ -> failwith "empty list you fool!"
            | x::xs, 0 -> (x, (List.rev acc) @ xs)
            | x::xs, n -> rmAt (x::acc) xs (n - 1)
    rmAt [] xs n
(*[/omit]*)

(*[omit:(Solution 2)]*)
// using problem 17
let removeAt' n xs = 
    let front,back = split xs n
    List.head back, front @ List.tail back
(*[/omit]*)
// [/snippet]// [snippet: Ninety-Nine F# Problems - Problems 21 - 28 - Lists again]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
// [/snippet]

// [snippet: (*) Problem 21 : Insert an element at a given position into a list.]
/// Example: 
/// * (insert-at 'alfa '(a b c d) 2)
/// (A ALFA B C D)
///  
/// Example in F#: 
/// 
/// > insertAt 'X' (List.ofSeq "abcd") 2;;
/// val it : char list = ['a'; 'X'; 'b'; 'c'; 'd']

(*[omit:(Solution 1)]*)
let insertAt x xs n =
    let rec insAt acc xs n =
        match xs, n with
            | [], 1 -> List.rev (x::acc)
            | [], _ -> failwith "Empty list you fool!"
            | xs, 1 -> List.rev (x::acc) @ xs
            | x::xs, n -> insAt (x::acc) xs (n - 1)
    insAt [] xs n
(*[/omit]*)

(*[omit:(Solution 2)]*)
let rec insertAt' x xs n =
    match xs, n with
        | [], 1 -> [x]
        | [], _ -> failwith "Empty list you fool!"
        | _, 1 -> x::xs
        | y::ys, n -> y::insertAt' x ys (n - 1)
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 22 : Create a list containing all integers within a given range.]
/// Example: 
/// * (range 4 9)
/// (4 5 6 7 8 9)
///  
/// Example in F#: 
/// 
/// > range 4 9;;
/// val it : int list = [4; 5; 6; 7; 8; 9]

(*[omit:(Solution)]*)
let range a b = [a..b]
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 23 : Extract a given number of randomly selected elements from a list.]
/// Example: 
/// * (rnd-select '(a b c d e f g h) 3)
/// (E D A)
///  
/// Example in F#: 
/// 
/// > rnd_select "abcdefgh" 3;;
/// val it : seq<char> = seq ['e'; 'a'; 'h']

(*[omit:(Solution)]*)
let rnd_select xs n = 
    let rndSeq = let r = new System.Random() in seq { while true do yield r.Next() }
    xs |> Seq.zip rndSeq |> Seq.sortBy fst |> Seq.map snd |> Seq.take n
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 24 : Lotto: Draw N different random numbers from the set 1..M.]
/// Example: 
/// * (rnd-select 6 49)
/// (23 1 17 33 21 37)
///  
/// Example in F#: 
/// 
/// > diff_select 6 49;;
/// val it : int list = [27; 20; 22; 9; 15; 29]

// using problem 23
(*[omit:(Solution)]*)
let diff_select n m = rnd_select (seq { 1 .. m }) n |> List.ofSeq
(*[/omit]*)

(*[omit:(Solution)]*)
let diff_select' n m = 
    let rndSeq = let r = new System.Random() in Seq.initInfinite(ignore >> r.Next ) 
    seq { 1 .. m } |> Seq.zip rndSeq |> Seq.sortBy fst |> Seq.map snd |> Seq.take n |> List.ofSeq
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 25 : Generate a random permutation of the elements of a list.]
/// Example: 
/// * (rnd-permu '(a b c d e f))
/// (B A D C E F)
///  
/// Example in F#: 
/// 
/// > rnd_permu <| List.ofSeq "abcdef";;
/// val it : char list = ['b'; 'c'; 'd'; 'f'; 'e'; 'a']

(*[omit:(Solution 1)]*)
// using problem 23
let rnd_permu xs = List.length xs |> rnd_select xs
(*[/omit]*)

(*[omit:(Solution 2)]*)
let rnd_permu' xs = 
    let rec rndSeq = 
        let r = new System.Random()
        seq { while true do yield r.Next() }
    xs |> Seq.zip rndSeq |> Seq.sortBy fst |> Seq.map snd |> List.ofSeq
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 26 : Generate the combinations of K distinct objects chosen from the N elements of a list.]
/// In how many ways can a committee of 3 be chosen from a group of 12 people? We all know that 
/// there are C(12,3) = 220 possibilities (C(N,K) denotes the well-known binomial coefficients). For 
/// pure mathematicians, this result may be great. But we want to really generate all the 
/// possibilities in a list.
///  
/// Example: 
/// * (combinations 3 '(a b c d e f))
/// ((A B C) (A B D) (A B E) ... )
///  
/// Example in F#: 
/// 
/// > combinations 3 ['a' .. 'f'];;
/// val it : char list list =
///   [['a'; 'b'; 'c']; ['a'; 'b'; 'd']; ['a'; 'b'; 'e']; ['a'; 'b'; 'f'];
///    ['a'; 'c'; 'd']; ['a'; 'c'; 'e']; ['a'; 'c'; 'f']; ['a'; 'd'; 'e'];
///    ['a'; 'd'; 'f']; ['a'; 'e'; 'f']; ['b'; 'c'; 'd']; ['b'; 'c'; 'e'];
///    ['b'; 'c'; 'f']; ['b'; 'd'; 'e']; ['b'; 'd'; 'f']; ['b'; 'e'; 'f'];
///    ['c'; 'd'; 'e']; ['c'; 'd'; 'f']; ['c'; 'e'; 'f']; ['d'; 'e'; 'f']] 

(*[omit:(Solution 1)]*)
// as a bonus you get the powerset of xs with combinations 0 xs
let rec combinations n xs =
    match xs, n with
        | [],_ -> [[]]
        | xs, 1 -> [for x in xs do yield [x]]
        | x::xs, n -> 
            [for ys in combinations (n-1) xs do
                yield x::ys
             if List.length xs > n then
                yield! combinations n xs
             else
                yield xs]
(*[/omit]*)

(*[omit:(Solution 2)]*)
let rec combinations' n xs =
    let rec tails = function
        | [] -> [[]]
        | _::ys as xs -> xs::tails ys
    match xs, n with
        | _, 0 -> [[]]
        | xs, n ->
            [ for tail in tails xs do
                match tail with
                    | [] -> ()
                    | y::xs' ->
                        for ys in combinations' (n - 1) xs' do
                            yield y::ys ]
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 27 : Group the elements of a set into disjoint subsets.] 
/// a) In how many ways can a group of 9 people work in 3 disjoint subgroups of 2, 3 
/// and 4 persons? Write a function that generates all the possibilities and returns them 
/// in a list.
///  
/// Example: 
/// * (group3 '(aldo beat carla david evi flip gary hugo ida))
/// ( ( (ALDO BEAT) (CARLA DAVID EVI) (FLIP GARY HUGO IDA) )
/// ... )
///  
/// b) Generalize the above predicate in a way that we can specify a list of group sizes 
/// and the predicate will return a list of groups.
///  
/// Example: 
/// * (group '(aldo beat carla david evi flip gary hugo ida) '(2 2 5))
/// ( ( (ALDO BEAT) (CARLA DAVID) (EVI FLIP GARY HUGO IDA) )
/// ... )
///  
/// Note that we do not want permutations of the group members; i.e. ((ALDO BEAT) ...) 
/// is the same solution as ((BEAT ALDO) ...). However, we make a difference between 
/// ((ALDO BEAT) (CARLA DAVID) ...) and ((CARLA DAVID) (ALDO BEAT) ...).
///  
/// You may find more about this combinatorial problem in a good book on discrete 
/// mathematics under the term "multinomial coefficients".
///  
/// Example in F#: 
/// 
/// > group [2;3;4] ["aldo";"beat";"carla";"david";"evi";"flip";"gary";"hugo";"ida"];;
/// val it : string list list list =
///   [[["aldo"; "beat"]; ["carla"; "david"; "evi"];
///     ["flip"; "gary"; "hugo"; "ida"]];...]
/// (altogether 1260 solutions)
///  
/// > group [2;2;5] ["aldo";"beat";"carla";"david";"evi";"flip";"gary";"hugo";"ida"];;
/// val it : string list list list =
///   [[["aldo"; "beat"]; ["carla"; "david"];
///     ["evi"; "flip"; "gary"; "hugo"; "ida"]];...]
/// (altogether 756 solutions)

(*[omit:(Solution)]*)
let rec group ns xs = 
    let rec combination n xs = 
        match n,xs with
            | 0, xs -> [([], xs)]
            | _, [] -> []
            | n, x::xs -> 
                let ts = [ for ys, zs in combination (n-1) xs do yield (x::ys, zs)]
                let ds = [ for ys, zs in combination n xs do yield (ys, x::zs)]
                ts @ ds
    match ns,xs with
        | [], _ -> [[]]
        | n::ns, xs ->
            [ for g, rs in combination n xs do
                for gs in group ns rs do
                    yield g::gs ]
(*[/omit]*)
// [/snippet]
    
// [snippet: (**) Problem 28 : Sorting a list of lists according to length of sublists]
/// a) We suppose that a list contains elements that are lists themselves. The objective 
/// is to sort the elements of this list according to their length. E.g. short lists first,
/// longer lists later, or vice versa.
///  
/// Example: 
/// * (lsort '((a b c) (d e) (f g h) (d e) (i j k l) (m n) (o)))
/// ((O) (D E) (D E) (M N) (A B C) (F G H) (I J K L))
///  
/// Example in F#: 
/// 
/// > lsort ["abc";"de";"fgh";"de";"ijkl";"mn";"o"];;
/// val it : string list = ["o"; "de"; "de"; "mn"; "abc"; "fgh"; "ijkl"]
///
/// b) Again; we suppose that a list contains elements that are lists themselves. But this 
/// time the objective is to sort the elements of this list according to their length 
/// frequency; i.e.; in the default; where sorting is done ascendingly; lists with rare 
/// lengths are placed first; others with a more frequent length come later.
///  
/// Example: 
/// * (lfsort '((a b c) (d e) (f g h) (d e) (i j k l) (m n) (o)))
/// ((i j k l) (o) (a b c) (f g h) (d e) (d e) (m n))
///  
/// Example in F#: 
/// 
/// > lfsort ["abc"; "de"; "fgh"; "de"; "ijkl"; "mn"; "o"];;
/// val it : string list = ["ijkl"; "o"; "abc"; "fgh"; "de"; "de"; "mn"]

(*[omit:(Solution)]*)
let lsort xss = xss |> List.sortBy Seq.length

let lfsort xss = xss |> Seq.groupBy (Seq.length) |> Seq.sortBy (snd >> Seq.length) |> Seq.collect snd |> List.ofSeq
(*[/omit]*)
// [/snippet]
// [snippet: Ninety-Nine F# Problems - Problems 31 - 41 - Arithmetic]
/// Ninety-Nine F# Problems - Problems 31 - 41 
///
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
// [/snippet]

// [snippet: (**) Problem 31 : Determine whether a given integer number is prime.]
/// Example: 
/// * (is-prime 7)
/// T
///  
/// Example in F#: 
/// 
/// > isPrime 7;;
/// val it : bool = true

(*[omit:(Solution 1)]*)
//naive solution
let isPrime n = 
    let sqrtn n = int <| sqrt (float n)
    seq { 2 .. sqrtn n } |> Seq.exists(fun i -> n % i = 0) |> not
(*[/omit]*)

(*[omit:(Solution 2)]*)
// Miller-Rabin primality test
open System.Numerics

let pow' mul sq x' n' = 
    let rec f x n y = 
        if n = 1I then
            mul x y
        else
            let (q,r) = BigInteger.DivRem(n, 2I)
            let x2 = sq x
            if r = 0I then
                f x2 q y
            else
                f x2 q (mul x y)
    f x' n' 1I
        
let mulMod (a :bigint) b c = (b * c) % a
let squareMod (a :bigint) b = (b * b) % a
let powMod m = pow' (mulMod m) (squareMod m)
let iterate f = Seq.unfold(fun x -> let fx = f x in Some(x,fx))

///See: http://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
let millerRabinPrimality n a =
    let find2km n = 
        let rec f k m = 
            let (q,r) = BigInteger.DivRem(m, 2I)
            if r = 1I then
                (k,m)
            else
                f (k+1I) q
        f 0I n
    let n' = n - 1I
    let iter = Seq.tryPick(fun x -> if x = 1I then Some(false) elif x = n' then Some(true) else None)
    let (k,m) = find2km n'
    let b0 = powMod n a m

    match (a,n) with
        | _ when a <= 1I && a >= n' -> 
            failwith (sprintf "millerRabinPrimality: a out of range (%A for %A)" a n)
        | _ when b0 = 1I || b0 = n' -> true
        | _  -> b0 
                 |> iterate (squareMod n) 
                 |> Seq.take(int k)
                 |> Seq.skip 1 
                 |> iter 
                 |> Option.exists id 

///For Miller-Rabin the witnesses need to be selected at random from the interval [2, n - 2]. 
///More witnesses => better accuracy of the test.
///Also, remember that if Miller-Rabin returns true, then the number is _probable_ prime. 
///If it returns false the number is composite.
let isPrimeW witnesses = function
    | n when n < 2I -> false
    | n when n = 2I -> true
    | n when n = 3I -> true
    | n when n % 2I = 0I -> false
    | n             -> witnesses |> Seq.forall(millerRabinPrimality n)

// let isPrime' = isPrimeW [2I;3I] // Two witnesses
// let p = pown 2I 4423 - 1I // 20th Mersenne prime. 1,332 digits
// isPrime' p |> printfn "%b";;
// Real: 00:00:03.184, CPU: 00:00:03.104, GC gen0: 12, gen1: 0, gen2: 0
// val it : bool = true
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 32 : Determine the greatest common divisor of two positive integer numbers. Use Euclid's algorithm.]
/// Example: 
/// * (gcd 36 63)
/// 9
///  
/// Example in F#: 
/// 
/// > [gcd 36 63; gcd (-3) (-6); gcd (-3) 6];;
/// val it : int list = [9; 3; 3]

(*[omit:(Solution)]*)
let rec gcd a b =
    if b = 0 then
        abs a
    else
        gcd b (a % b)
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 33 : Determine whether two positive integer numbers are coprime.]
/// Two numbers are coprime if their greatest common divisor equals 1.
///  
/// Example: 
/// * (coprime 35 64)
/// T
///  
/// Example in F#: 
/// 
/// > coprime 35 64;;
/// val it : bool = true

(*[omit:(Solution)]*)
// using problem 32
let coprime a b = gcd a b = 1
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 34 : Calculate Euler's totient function phi(m).]
/// Euler's so-called totient function phi(m) is defined as the number of 
/// positive integers r (1 <= r < m) that are coprime to m.
///  
/// Example: m = 10: r = 1,3,7,9; thus phi(m) = 4. Note the special case: phi(1) = 1.
///  
/// Example: 
/// * (totient-phi 10)
/// 4
///  
/// Example in F#: 
/// 
/// > totient 10;;
/// val it : int = 4

(*[omit:(Solution)]*)
// naive implementation. For a better solution see problem 37
let totient n = seq { 1 .. n - 1} |> Seq.filter (gcd n >> (=) 1) |> Seq.length
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 35 : Determine the prime factors of a given positive integer.]
/// Construct a flat list containing the prime factors in ascending order.
///  
/// Example: 
/// * (prime-factors 315)
/// (3 3 5 7)
///  
/// Example in F#: 
/// 
/// > primeFactors 315;;
/// val it : int list = [3; 3; 5; 7]

(*[omit:(Solution)]*)
let primeFactors n =
    let sqrtn n = int <| sqrt (float n)
    let get n =
        let sq = sqrtn n
        // this can be made faster by using a prime generator like this one : 
        // https://github.com/paks/ProjectEuler/tree/master/Euler/Primegen
        seq { yield 2; yield! seq {3 .. 2 .. sq} } |> Seq.tryFind (fun x -> n % x = 0) 
    let divSeq = n |> Seq.unfold(fun x ->
        if x = 1 then
            None
        else
            match get x with
                | None -> Some(x, 1) // x it's prime
                | Some(divisor) -> Some(divisor, x/divisor))
    divSeq |> List.ofSeq
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 36 : Determine the prime factors of a given positive integer.]
/// 
/// Construct a list containing the prime factors and their multiplicity. 
/// 
/// Example: 
/// * (prime-factors-mult 315)
/// ((3 2) (5 1) (7 1))
///  
/// Example in F#: 
/// 
/// > primeFactorsMult 315;;
/// [(3,2);(5,1);(7,1)]

(*[omit:(Solution)]*)
// using problem 35
let primeFactorsMult n =
    let sqrtn n = int <| sqrt (float n)
    let get n =
        let sq = sqrtn n
        // this can be made faster by using a prime generator like this one : 
        // https://github.com/paks/ProjectEuler/tree/master/Euler/Primegen
        seq { yield 2; yield! seq {3 .. 2 .. sq} } |> Seq.tryFind (fun x -> n % x = 0) 
    let divSeq = n |> Seq.unfold(fun x ->
        if x = 1 then
            None
        else
            match get x with
                | None -> Some(x, 1) // x it's prime
                | Some(divisor) -> Some(divisor, x/divisor))
    divSeq |> Seq.countBy id |> List.ofSeq
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 37 : Calculate Euler's totient function phi(m) (improved).]
/// See problem 34 for the definition of Euler's totient function. If the list of the prime 
/// factors of a number m is known in the form of problem 36 then the function phi(m) 
/// can be efficiently calculated as follows: Let ((p1 m1) (p2 m2) (p3 m3) ...) be the list of 
/// prime factors (and their multiplicities) of a given number m. Then phi(m) can be 
/// calculated with the following formula:
///  phi(m) = (p1 - 1) * p1 ** (m1 - 1) + 
///          (p2 - 1) * p2 ** (m2 - 1) + 
///          (p3 - 1) * p3 ** (m3 - 1) + ...
///  
/// Note that a ** b stands for the b'th power of a. 
/// 
/// Note: Actually, the official problems show this as a sum, but it should be a product.
/// > phi 10;;
/// val it : int = 4

(*[omit:(Solution)]*)
// using problem 36
let phi = primeFactorsMult >> Seq.fold(fun acc (p,m) -> (p - 1) * pown p (m - 1) * acc) 1
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 38 : Compare the two methods of calculating Euler's totient function.]
/// Use the solutions of problems 34 and 37 to compare the algorithms. Take the 
/// number of reductions as a measure for efficiency. Try to calculate phi(10090) as an 
/// example.
///  
/// (no solution required) 
/// 
// [/snippet]

// [snippet: (*) Problem 39 : A list of prime numbers.]
/// Given a range of integers by its lower and upper limit, construct a list of all prime numbers
/// in that range.
///  
/// Example in F#: 
/// 
/// > primesR 10 20;;
/// val it : int list = [11; 13; 17; 19]

(*[omit:(Solution)]*)
// using problem 31
let primeR a b = seq { a .. b } |> Seq.filter isPrime |> List.ofSeq
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 40 : Goldbach's conjecture.]
/// Goldbach's conjecture says that every positive even number greater than 2 is the 
/// sum of two prime numbers. Example: 28 = 5 + 23. It is one of the most famous facts 
/// in number theory that has not been proved to be correct in the general case. It has 
/// been numerically confirmed up to very large numbers (much larger than we can go 
/// with our Prolog system). Write a predicate to find the two prime numbers that sum up 
/// to a given even integer.
///  
/// Example: 
/// * (goldbach 28)
/// (5 23)
///  
/// Example in F#: 
/// 
/// *goldbach 28
/// val it : int * int = (5, 23)

(*[omit:(Solution)]*)
// using problem 31. Very slow on big numbers due to the implementation of primeR. To speed this up use a prime generator.
let goldbach n =
    let primes = primeR 2 n |> Array.ofList
    let rec findPairSum (arr: int array) front back =
        let sum = arr.[front] + arr.[back]
        match compare sum n with
            | -1 -> findPairSum arr (front + 1) back
            |  0 -> Some(arr.[front] , arr.[back])
            |  1 -> findPairSum arr front (back - 1)
            |  _ -> failwith "not possible"
    Option.get <| findPairSum primes 0 (primes.Length - 1)
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 41 : Given a range of integers by its lower and upper limit, print a list of all even numbers and their Goldbach composition.]
/// In most cases, if an even number is written as the sum of two prime numbers, one of 
/// them is very small. Very rarely, the primes are both bigger than say 50. Try to find 
/// out how many such cases there are in the range 2..3000.
///  
/// Example: 
/// * (goldbach-list 9 20)
/// 10 = 3 + 7
/// 12 = 5 + 7
/// 14 = 3 + 11
/// 16 = 3 + 13
/// 18 = 5 + 13
/// 20 = 3 + 17
/// * (goldbach-list 1 2000 50)
/// 992 = 73 + 919
/// 1382 = 61 + 1321
/// 1856 = 67 + 1789
/// 1928 = 61 + 1867
///  
/// Example in F#: 
/// 
/// > goldbachList 9 20;;
/// val it : (int * int) list =
///   [(3, 7); (5, 7); (3, 11); (3, 13); (5, 13); (3, 17)]
/// > goldbachList' 4 2000 50
/// val it : (int * int) list = [(73, 919); (61, 1321); (67, 1789); (61, 1867)]

(*[omit:(Solution)]*)
let goldbachList a b =
    let start = if a % 2 <> 0 then a + 1 else a
    seq { start .. 2 .. b } |> Seq.map goldbach |> List.ofSeq

let goldbachList' a b limit = goldbachList a b |> List.filter(fst >> (<) limit)
(*[/omit]*)
// [/snippet]
// [snippet: Ninety-Nine F# Problems - Problems 46 - 50 - Logic and Codes]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
// [/snippet]

// [snippet: (**) Problem 46 : Define Logical predicates]
/// Define predicates and/2, or/2, nand/2, nor/2, xor/2, impl/2 and equ/2 (for logical 
/// equivalence) which succeed or fail according to the result of their respective 
/// operations; e.g. and(A,B) will succeed, if and only if both A and B succeed.
///  
/// A logical expression in two variables can then be written as in the following example: 
/// and(or(A,B),nand(A,B)).
///  
/// Now, write a predicate table/3 which prints the truth table of a given logical 
/// expression in two variables.
///  
/// Example: 
/// (table A B (and A (or A B)))
/// true true true
/// true fail true
/// fail true fail
/// fail fail fail
///  
/// Example in F#: 
/// 
/// > table (fun a b -> (and' a (or' a b)));;
/// true true true
/// true false true
/// false true false
/// false false false
/// val it : unit = ()

(*[omit:(Solution)]*)
let and' = (&&)

let or'  = (||)

let nand a b = not <| and' a b

let nor a b = not <| or' a b

let xor a b = if a <> b then true else false

let impl a b = compare a b |> (<>) 1

let eq = (=)

let table expr = 
    let inputs = [ (true, true); (true, false); (false, true); (false, false) ]
    inputs |> Seq.iter (fun (a,b) -> printfn "%b %b %b" a b (expr a b))
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 47 : Truth tables for logical expressions (2).]
/// Continue problem P46 by defining and/2, or/2, etc as being operators. This allows to write 
/// the logical expression in the more natural way, as in the example: A and (A or not B). 
/// Define operator precedence as usual; i.e. as in Java.
///  
/// Example: 
/// * (table A B (A and (A or not B)))
/// true true true
/// true fail true
/// fail true fail
/// fail fail fail
///  
/// Example in F#: 
/// 
/// > table2 (fun a b -> a && (a || not b));;
/// true true true
/// true false true
/// false true false
/// false false false
/// val it : unit = ()

(*[omit:(Solution)]*)
// let's use the F# built-in operateros plus:

// xor
let (&|) a b = if a <> b then true else false

// nand
let (^&&) a b = not <| a && b

// nor
let (^||) a b = not <| a || b

// impl
let (|->) a b = compare a b |> (<>) 1

// same as problem 46
let table2 expr = 
    let inputs = [ (true, true); (true, false); (false, true); (false, false) ]
    inputs |> Seq.iter (fun (a,b) -> printfn "%b %b %b" a b (expr a b))
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 48 : Truth tables for logical expressions (3).]
/// Generalize problem P47 in such a way that the logical expression may contain any 
/// number of logical variables. Define table/2 in a way that table(List,Expr) prints the 
/// truth table for the expression Expr, which contains the logical variables enumerated 
/// in List.
///  
/// Example: 
/// * (table (A,B,C) (A and (B or C) equ A and B or A and C))
/// true true true true
/// true true fail true
/// true fail true true
/// true fail fail true
/// fail true true true
/// fail true fail true
/// fail fail true true
/// fail fail fail true
///  
/// Example in F#: 
/// 
/// > tablen 3 (fun [a;b;c] -> a && (b || c) = a && b || a && c)
/// warning FS0025: Incomplete pattern matches on this expression. ...
/// True True True true
/// False True True false
/// True False True true
/// False False True false
/// True True False true
/// False True False false
/// True False False false
/// False False False false
/// val it : unit = ()

(*[omit:(Solution)]*)
let tablen n expr =
    let replicate n xs = 
        let rec repl acc n =
            match n with
                | 0 -> acc
                | n -> 
                    let acc' = acc |> List.collect(fun ys -> xs |> List.map(fun x -> x::ys))
                    repl acc' (n-1)
        repl [[]] n 
 
    let values = replicate n [true; false]
    let toString bs = System.String.Join(" ", Array.ofList (bs |> List.map string))
    values |> Seq.iter(fun bs -> printfn "%s %b" (bs |> toString) (expr bs))
(*[/omit]*)
// [/snippet]


// [snippet: (**) Problem 49 : Gray codes.]
/// An n-bit Gray code is a sequence of n-bit strings constructed according to certain rules.
/// For example,
///
/// n = 1: C(1) = ['0','1'].
/// n = 2: C(2) = ['00','01','11','10'].
/// n = 3: C(3) = ['000','001','011','010',´110´,´111´,´101´,´100´].
///  
/// Find out the construction rules and write a predicate with the following specification:
///  % gray(N,C) :- C is the N-bit Gray code
///  
/// Can you apply the method of "result caching" in order to make the predicate more efficient, 
/// when it is to be used repeatedly?
///  
/// Example in F#: 
/// 
/// P49> gray 3
/// ["000","001","011","010","110","111","101","100"]

(*[omit:(Solution)]*)
// The rules to contruct gray codes can be found here : http://en.wikipedia.org/wiki/Gray_code
let rec gray = function
    | 0 -> [""]
    | n -> 
        let prev = gray (n - 1)
        (prev |> List.map ((+) "0")) @ (prev |> List.rev |> List.map((+) "1"))
(*[/omit]*)
// [/snippet]
   
// [snippet: (***) Problem 50 : Huffman codes.]
/// We suppose a set of symbols with their frequencies, given as a list of fr(S,F) terms. 
/// Example: [fr(a,45),fr(b,13),fr(c,12),fr(d,16),fr(e,9),fr(f,5)]. Our objective is to 
/// construct /// a list hc(S,C) terms, where C is the Huffman code word for the symbol 
/// S. In our example, the result could be Hs = [hc(a,'0'), hc(b,'101'), hc(c,'100'), 
/// hc(d,'111'), hc(e,'1101'), hc(f,'1100')] [hc(a,'01'),...etc.]. The task shall be 
/// performed by the predicate huffman/2 defined as follows:
/// 
///  % huffman(Fs,Hs) :- Hs is the Huffman code table for the frequency table Fs
///  
/// Example in F#: 
/// 
/// > huffman [('a',45);('b',13);('c',12);('d',16);('e',9);('f',5)];;
/// val it : (char * string) list =
///   [('a', "0"); ('b', "101"); ('c', "100"); ('d', "111"); ('e', "1101");
///    ('f', "1100")]

(*[omit:(Solution)]*)
// First we create a representation of the Huffman tree
type 'a HuffmanTree = Node of int (*frecuency*) * 'a (* left *) HuffmanTree * 'a (* right *) HuffmanTree | Leaf of int * 'a (* term *)

// Auxiliary function to get the frecuency
let frecuency = function
    | Leaf (frec, _) -> frec
    | Node(frec, _, _) -> frec

// Once we have build the Huffman tree, we can use this function to assing the codes
// nodes to the left get a '0'. Nodes to the right get a '1'.
let encode tree =
    let rec enc code tree cont =
        match tree with
            | Leaf (_, a) -> cont [(a, code)]
            | Node(_, lt, rt) ->
                enc (code + "0") lt <| fun ltacc -> enc (code + "1") rt <| fun rtacc -> cont (ltacc @ rtacc)
    enc "" tree id

// The algorithm is explained here: http://en.wikipedia.org/wiki/Huffman_coding
// The implementation below uses lists. For better performance use a priority queue.
// This is how it works. First we transform the list of terms and frecuencies into a list of Leafs (6).
// Then, before anything happpens, we sort the list to place the terms with the lowest frecuency
// at the head of the List (1) (this is where a priority queue would shine). 
// Otherwise, we combine the first two elements into a Node with the combined frecuency of the two nodes (4). 
// We add the node to the list and try again (5). Eventualy the list is reduced to 
// one term and we're done constructing the tree (2). Once we have the tree, we just need to encode it (7).
let huffman symbols =
    let rec createTree tree = 
        let xs = tree |> List.sortBy frecuency (* 1 *)
        match xs with
            | [] -> failwith "Empty list"
            | [x] -> x (* 2 *)
            | x::y::xs -> (* 3 *)
                let ht = Node(frecuency x + frecuency y, x , y) (* 4 *)
                createTree (ht::xs) (* 5 *)
    let ht = symbols 
             |> List.map(fun (a,f) -> Leaf (f,a)) (* 6 *)
             |> createTree
    encode ht |> List.sortBy(fst) (* 7 *)
(*[/omit]*)
// [/snippet]// [snippet: Ninety-Nine F# Problems - Problems 54 - 60 - Binary trees]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
///
///  
/// 
/// A binary tree is either empty or it is composed of a root element and two successors, 
/// which are binary trees themselves.
/// 
///                              (a)
///                             /   \
///                            (b)  (c)
///                           /  \    \
///                         (d)  (e)  (f)
///
/// In F#, we can characterize binary trees with a type definition: 
/// 

type 'a Tree = Empty | Branch of 'a * 'a Tree * 'a Tree

///
/// This says that a Tree of type a consists of either an Empty node, or a Branch containing one 
/// value of type a with exactly two subtrees of type a.
///  
/// Given this definition, the tree in the diagram above would be represented as: 
/// 

let tree1 = Branch ('a', Branch ('b', Branch ('d', Empty, Empty),
                               Branch ('e', Empty, Empty)),
                         Branch ('c', Empty,
                               Branch ('f', Branch ('g', Empty, Empty),
                                           Empty))) 

/// Since a "leaf" node is a branch with two empty subtrees, it can be useful to define a 
/// shorthand function:

let leaf x = Branch (x, Empty, Empty) 

/// Then the tree diagram above could be expressed more simply as: 

let tree1' = Branch ('a', Branch ('b', leaf 'd',
                               leaf 'e'),
                          Branch ('c', Empty,
                               Branch ('f', leaf 'g',
                                           Empty)))
/// Other examples of binary trees: 
/// 
/// -- A binary tree consisting of a root node only
let tree2 = Branch ('a', Empty, Empty)
///  
/// -- An empty binary tree
let tree3 = Empty
///  
/// -- A tree of integers
let tree4 = Branch (1, Branch (2, Empty, Branch (4, Empty, Empty)),
                       Branch (2, Empty, Empty))
// [/snippet]

// [snippet: (*) Problem 54A : Check whether a given term represents a binary tree.]
/// In Prolog or Lisp, one writes a predicate to do this. 
/// 
/// Example in Lisp: 
/// * (istree (a (b nil nil) nil))
/// T
/// * (istree (a (b nil nil)))
/// NIL
///  
/// Non-solution: 
/// F#'s type system ensures that all terms of type 'a Tree are binary trees: it is just not 
//  possible to construct an invalid tree with this type. Hence, it is redundant to introduce 
/// a predicate to check this property: it would always return True
// [/snippet]

// [snippet: (**) Problem 55 : Construct completely balanced binary trees]
/// In a completely balanced binary tree, the following property holds for every node: 
/// The number of nodes in its left subtree and the number of nodes in its right subtree 
/// are almost equal, which means their difference is not greater than one.
///  
/// Write a function cbal-tree to construct completely balanced binary trees for a given 
/// number of nodes. The predicate should generate all solutions via backtracking. Put 
/// the letter 'x' as information into all nodes of the tree.
///  
/// Example: 
/// * cbal-tree(4,T).
/// T = t(x, t(x, nil, nil), t(x, nil, t(x, nil, nil))) ;
/// T = t(x, t(x, nil, nil), t(x, t(x, nil, nil), nil)) ;
/// etc......No
///  
/// Example in F#, whitespace and "comment diagrams" added for clarity and exposition:
///  
/// > cbalTree 4;;
/// val trees : char Tree list =
/// [
///    permutation 1
///        x
///       / \
///      x   x
///           \
///            x
/// Branch ('x', Branch ('x', Empty, Empty),
///              Branch ('x', Empty,
///                        Branch ('x', Empty, Empty)));
///  
///    permutation 2
///        x
///       / \
///      x   x
///         /
///        x
/// Branch ('x', Branch ('x', Empty, Empty),
///              Branch ('x', Branch ('x', Empty, Empty),
///                        Empty));
///  
///    permutation 3
///        x
///       / \
///      x   x
///       \
///        x
/// Branch ('x', Branch ('x', Empty, 
///                           Branch ('x', Empty, Empty)),
///              Branch ('x', Empty, Empty));
///  
///    permutation 4
///        x
///       / \
///      x   x
///     /
///    x
/// Branch ('x', Branch ('x', Branch ('x', Empty, Empty),
///                        Empty), 
///              Branch ('x', Empty, Empty))
/// ]

(*[omit:(Solution 1)]*)
let rec cbalTree n =
    match n with
        | 0 -> [Empty]
        | n -> let q,r = let x = n - 1 in x / 2, x % 2 
               [ for i=q to q + r do
                    for lt in cbalTree i do
                       for rt in cbalTree (n - 1 - i) do
                          yield Branch('x', lt, rt) ]
(*[/omit]*)

(*[omit:(Solution 2)]*)
let nodes t = 
    let rec nodes' t cont = 
        match t with
            | Empty -> cont 0
            | Branch(_, lt, rt) -> 
                nodes' lt (fun nlt -> nodes' rt (fun nrt -> cont (1 + nlt + nrt)))
    nodes' t id

let rec allTrees n =
    match n with
        | 0 -> [Empty]
        | n ->
            [ for i=0 to n - 1 do
                 for lt in cbalTree i do
                    for rt in cbalTree (n - 1 - i) do
                       yield Branch('x', lt, rt) ]

let cbalTree' n = allTrees n |> List.filter(fun t -> 
                                                match t with
                                                    | Empty -> true
                                                    | Branch(_, lt, rt) -> abs (nodes lt - nodes rt) <= 1 )
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 56 : Symmetric binary trees]
/// Let us call a binary tree symmetric if you can draw a vertical line through the root 
/// node and then the right subtree is the mirror image of the left subtree. Write a 
/// predicate symmetric/1 to check whether a given binary tree is symmetric. Hint: Write 
/// a predicate mirror/2 first to check whether one tree is the mirror image of another. 
/// We are only interested in the structure, not in the contents of the nodes.
///  
/// Example in F#: 
/// 
/// > symmetric <| Branch ('x', Branch ('x', Empty, Empty), Empty);;
/// val it : bool = false
/// > symmetric <| Branch ('x', Branch ('x', Empty, Empty), Branch ('x', Empty, Empty))
/// val it : bool = true

(*[omit:(Solution)]*)
let symmetric tree =
    let rec mirror t1 t2 cont =
        match t1,t2 with
            | Empty,Empty -> cont true
            | Empty, Branch _ -> cont false
            | Branch _, Empty -> cont false
            | Branch (_, lt1, rt1), Branch (_, lt2, rt2) -> 
                mirror lt1 rt2 (fun isMirrorLeft -> mirror rt1 lt2 (fun isMirrorRight -> cont (isMirrorLeft && isMirrorRight)))
    match tree with
        | Empty -> true
        | Branch (_,lt, rt) -> mirror lt rt id
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 57 : Binary search trees (dictionaries)]
/// Use the predicate add/3, developed in chapter 4 of the course, to write a predicate 
/// to construct a binary search tree from a list of integer numbers.
///  
/// Example: 
/// * construct([3,2,5,7,1],T).
/// T = t(3, t(2, t(1, nil, nil), nil), t(5, nil, t(7, nil, nil)))
///  
/// Then use this predicate to test the solution of the problem P56. 
/// 
/// Example: 
/// * test-symmetric([5,3,18,1,4,12,21]).
/// Yes
/// * test-symmetric([3,2,5,7,4]).
/// No
///  
/// Example in F#: 
/// 
/// > construct [3; 2; 5; 7; 1]
/// val it : int Tree =
///   Branch (3,Branch (2,Branch (1,Empty,Empty),Empty),
///             Branch (5,Empty,Branch (7,Empty,Empty)))
/// > [5; 3; 18; 1; 4; 12; 21] |> construct |> symmetric;;
/// val it : bool = true
/// > [3; 2; 5; 7; 1] |> construct |> symmetric;;
/// val it : bool = true

(*[omit:(Solution)]*)
let insert x tree = 
    let rec insert' t cont =
        match t with
            | Empty -> cont <| Branch(x, Empty, Empty)
            | Branch(y, lt, rt) as t ->
                if x < y then
                    insert' lt <| fun lt' -> cont <| Branch(y, lt', rt)
                elif x > y then
                    insert' rt <| fun rt' -> cont <| Branch(y, lt, rt')
                else
                    t
    insert' tree id

let construct xs = xs |> List.fold(fun tree x -> insert x tree) Empty
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 58 : Generate-and-test paradigm]
/// Apply the generate-and-test paradigm to construct all symmetric, completely balanced 
/// binary trees with a given number of nodes.
///  
/// Example: 
/// * sym-cbal-trees(5,Ts).
/// Ts = [t(x, t(x, nil, t(x, nil, nil)), t(x, t(x, nil, nil), nil)), 
///       t(x, t(x, t(x, nil, nil), nil), t(x, nil, t(x, nil, nil)))] 
///  
/// Example in F#: 
/// 
/// > symCbalTrees 5;;
/// val it : char Tree list =
///   [Branch
///      ('x',Branch ('x',Empty,Branch ('x',Empty,Empty)),
///       Branch ('x',Branch ('x',Empty,Empty),Empty));
///    Branch
///      ('x',Branch ('x',Branch ('x',Empty,Empty),Empty),
///       Branch ('x',Empty,Branch ('x',Empty,Empty)))]

(*[omit:(Solution)]*)
let symCbalTrees = cbalTree >> List.filter symmetric
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 59 : Construct height-balanced binary trees]
/// In a height-balanced binary tree, the following property holds for every node: The 
/// height of its left subtree and the height of its right subtree are almost equal, 
/// which means their difference is not greater than one.
///  
/// Example: 
/// ?- hbal_tree(3,T).
/// T = t(x, t(x, t(x, nil, nil), t(x, nil, nil)), t(x, t(x, nil, nil), t(x, nil, nil))) ;
/// T = t(x, t(x, t(x, nil, nil), t(x, nil, nil)), t(x, t(x, nil, nil), nil)) ;
/// etc......No
///  
/// Example in F#: 
/// 
/// > hbalTree 'x' 3 |> Seq.take 4;;
/// val it : seq<char Tree> =
///   seq
///     [Branch
///        ('x',Branch ('x',Branch ('x',Empty,Empty),Branch ('x',Empty,Empty)),
///         Branch ('x',Branch ('x',Empty,Empty),Branch ('x',Empty,Empty)));
///      Branch
///        ('x',Branch ('x',Branch ('x',Empty,Empty),Branch ('x',Empty,Empty)),
///         Branch ('x',Branch ('x',Empty,Empty),Empty));
///      Branch
///        ('x',Branch ('x',Branch ('x',Empty,Empty),Branch ('x',Empty,Empty)),
///         Branch ('x',Empty,Branch ('x',Empty,Empty)));
///      Branch
///        ('x',Branch ('x',Branch ('x',Empty,Empty),Branch ('x',Empty,Empty)),
///         Branch ('x',Empty,Empty))]

(*[omit:(Solution)]*)
let hbalTree a height =
    let rec loop h cont = 
        match h with
            | 0 -> cont [Empty, 0]
            | 1 -> cont [Branch (a, Empty, Empty), 1]
            | _ -> loop (h-1) (fun lts ->
                       loop (h-2) (fun rts -> 
                       cont <| [let t = lts @ rts 
                                for (t1,h1) in t do
                                    for (t2,h2) in t do
                                        let ht = 1 + max h1 h2 
                                        if ht = h then
                                            yield Branch (a, t1, t2), ht] ))
    loop height id |> List.map fst
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 60 : Construct height-balanced binary trees with a given number of nodes]
/// Consider a height-balanced binary tree of height H. What is the maximum number of nodes 
/// it can contain?
/// Clearly, MaxN = 2**H - 1. However, what is the minimum number MinN? This question is more 
/// difficult. Try to find a recursive statement and turn it into a function minNodes that 
/// returns the minimum number of nodes in a height-balanced binary tree of height H. On the 
/// other hand, we might ask: what is the maximum height H a height-balanced binary tree with 
/// N nodes can have? Write a function maxHeight that computes this. 
///
/// Now, we can attack the main problem: construct all the height-balanced binary trees with a 
/// given nuber of nodes. Find out how many height-balanced trees exist for N = 15.
///  
/// Example in Prolog: 
/// ?- count_hbal_trees(15,C).
/// C = 1553
///  
/// Example in F#: 
/// 
/// > hbalTreeNodes 'x' 15 |> List.length;;
/// val it : int = 1553
/// > [0 .. 3] |> List.map (hbalTreeNodes 'x');;
/// val it : char Tree list list =
///   [[Empty]; [Branch ('x',Empty,Empty)];
///    [Branch ('x',Branch ('x',Empty,Empty),Empty);
///     Branch ('x',Empty,Branch ('x',Empty,Empty))];
///    [Branch ('x',Branch ('x',Empty,Empty),Branch ('x',Empty,Empty))]]

(*[omit:(Solution)]*)
let minNodes height = 
    let rec minNodes' h cont =
        match h with
        | 0 -> cont 0
        | 1 -> cont 1
        | _ -> minNodes' (h - 1) <| fun h1 -> minNodes' (h - 2) <| fun h2 -> cont <| 1 + h1 + h2
    minNodes' height id

let maxHeight nodes = 
    let rec loop n acc =
        match n with
            | 0 -> acc
            | _ -> loop (n >>> 1) (acc + 1)
    let fullHeight = loop nodes 0 // this is the height of a tree with full nodes
    let minNodesH1 = minNodes (fullHeight + 1)
    if nodes < minNodesH1 then
        fullHeight
    else
        fullHeight + 1

let numNodes tree = 
    let rec numNodes' tree cont =
        match tree with
            | Empty -> cont 0
            | Branch(_, lt , rt) ->
                numNodes' lt <| fun ln -> numNodes' rt <| fun rn -> cont <| 1 + ln + rn
    numNodes' tree id

let hbalTreeNodes x nodes = 
    let maxH = maxHeight nodes
    let minH = if maxH = 0 then 0 else maxH - 1
    [minH .. maxH] |> List.collect(fun n -> hbalTree x n) |> List.filter(fun t -> nodes = numNodes t)
(*[/omit]*)
// [/snippet]// [snippet: Ninety-Nine F# Problems - Problems 61 - 69 - Binary trees]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
///
///
/// Binary trees 
/// 
/// As defined in problem 54A. 

type 'a Tree = Empty | Branch of 'a * 'a Tree * 'a Tree

/// 
/// An example tree: 
/// 
let tree4 = Branch (1, Branch (2, Empty, Branch (4, Empty, Empty)),
                       Branch (2, Empty, Empty))
// [/snippet]


// [snippet: (*) Problem 61 : Count the leaves of a binary tree]
/// A leaf is a node with no successors. Write a predicate count_leaves/2 to count them.
///  
/// Example: 
/// % count_leaves(T,N) :- the binary tree T has N leaves
///  
/// Example in F#: 
/// 
/// > countLeaves tree4
/// val it : int = 2

(*[omit:(Solution)]*)
let foldTree branchF emptyV t =
    let rec loop t cont =
        match t with
        | Empty -> cont emptyV
        | Branch(x,left,right) -> loop left  (fun lacc -> 
                                  loop right (fun racc ->
                                  cont (branchF x lacc racc)))
    loop t id

let countLeaves tree = tree |> foldTree (fun _ lc rc -> 1 + lc + rc) 0
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 62 : Collect the internal nodes of a binary tree in a list]
/// An internal node of a binary tree has either one or two non-empty successors. Write a 
/// predicate internals/2 to collect them in a list.
///  
/// Example: 
/// % internals(T,S) :- S is the list of internal nodes of the binary tree T.
///  
/// Example in F#: 
/// 
/// >internals tree4;;
/// val it : int list = [1; 2]

(*[omit:(Solution)]*)
// using foldTree from problem 61
let insternals tree = tree |> foldTree (fun x (lc,lt) (rc,rt) -> if lt || rt  then ([x] @ lc @ rc ,true) else ([], true)) ([],false) |> fst
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 62B : Collect the nodes at a given level in a list]
/// A node of a binary tree is at level N if the path from the root to the node has 
/// length N-1. The root node is at level 1. Write a predicate atlevel/3 to collect 
/// all nodes at a given level in a list.
///  
/// Example: 
/// % atlevel(T,L,S) :- S is the list of nodes of the binary tree T at level L
///  
/// Example in F#: 
/// 
/// >atLevel tree4 2;;
/// val it : int list = [2,2]

(*[omit:(Solution)]*)
let atLevel tree level = 
    let rec loop l tree cont =
        match tree with
            | Empty -> cont []
            | Branch(x, lt , rt) -> 
                if l = level then
                    cont [x]
                else
                    loop (l + 1) lt (fun lacc -> loop (l + 1) rt (fun racc -> cont <| lacc @ racc))
    loop 1 tree id
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 63 : Construct a complete binary tree]
/// A complete binary tree with height H is defined as follows: 
/// • The levels 1,2,3,...,H-1 contain the maximum number of nodes (i.e 2**(i-1) at the 
///   level i)
/// • In level H, which may contain less than the maximum possible number of nodes, 
///   all the nodes are "left-adjusted". This means that in a levelorder tree traversal all 
///   internal nodes come first, the leaves come second, and empty successors (the 
///   nil's which are not really nodes!) come last.
///  
/// Particularly, complete binary trees are used as data structures (or addressing 
/// schemes) for heaps.
///  
/// We can assign an address number to each node in a complete binary tree by 
/// enumerating the nodes in level-order, starting at the root with number 1. For every 
/// node X with address A the following property holds: The address of X's left and right 
/// successors are 2*A and 2*A+1, respectively, if they exist. This fact can be used to 
/// elegantly construct a complete binary tree structure.
///  
/// Write a predicate complete_binary_tree/2. 
/// 
/// Example: 
/// % complete_binary_tree(N,T) :- T is a complete binary tree with N nodes.
///  
/// Example in F#: 
/// 
/// > completeBinaryTree 4
/// Branch ('x', Branch ('x', Branch ('x', Empty, Empty), Empty), 
///                                             Branch ('x', Empty, Empty))
///  
/// > isCompleteBinaryTree <|  Branch ('x', Branch ('x', Empty, Empty), 
///                                                    Branch ('x', Empty, Empty))
/// val it : bool = true

(*[omit:(Solution)]*)
let completeBinaryTree n = 
    let rec loop l cont =
        if l <= n then
            loop (2*l) (fun lt -> loop (2*l+1) (fun rt -> cont <| Branch ('x', lt, rt)))
        else
            cont Empty
    loop 1 id

let isCompleteBinaryTree tree =
    let rec loop level tree cont =
        match tree with
            | Empty -> cont ([], 0)
            | Branch(_, lt, rt) ->
                loop (2*level) lt (fun (ll,lc) -> loop (2*level+1) rt (fun (rl, rc) -> cont <| ([level] @ ll @ rl, 1 + lc + rc)))
    let levels, nodes = loop 1 tree (fun (ls,ns) -> List.sort ls, ns)
    levels |> Seq.zip (seq { 1 .. nodes }) |> Seq.forall(fun (a,b) -> a = b)
(*[/omit]*)
// [/snippet]
    

// [snippet: (**) Problem 64 : Layout a binary tree (1)]
/// Given a binary tree as the usual Prolog term t(X,L,R) (or nil). As a preparation for 
/// drawing the tree, a layout algorithm is required to determine the position of each 
/// node in a rectangular grid. Several layout methods are conceivable, one of them is 
/// shown in the illustration below:
///
///     1  2  3  4  5  6  7  8  9  10  11  12
/// 
/// 1                       (n)
///                       /             \
/// 2                 (k)                  (u)
///             /        \           /
/// 3     (c)            (m)   (p)
///      /     \                    \
/// 4  (a)         (h)                 (s)
///               /                   /
/// 5           (g)                (q)
///            /
/// 6        (e)
/// 
/// In this layout strategy, the position of a node v is obtained by the following two rules:
/// • x(v) is equal to the position of the node v in the inorder sequence 
/// • y(v) is equal to the depth of the node v in the tree 
/// 
/// Write a function to annotate each node of the tree with a position, where (1,1) in the 
/// top left corner or the rectangle bounding the drawn tree.
///  
/// Here is the example tree from the above illustration: 
/// 
let tree64 = Branch ('n',
                Branch ('k',
                        Branch ('c',
                                Branch ('a', Empty, Empty),
                                Branch ('h',
                                        Branch ('g',
                                                Branch ('e', Empty, Empty),
                                                Empty),
                                        Empty)
                                ),
                        Branch ('m', Empty, Empty)),
                Branch ('u',
                        Branch ('p',
                                Empty,
                                Branch ('s',
                                        Branch ('q', Empty, Empty),
                                        Empty)
                                ),
                        Empty
                ))
/// Example in F#: 
/// 
/// > layout tree64;;
/// val it : (char * (int * int)) Tree =
///   Branch
///     (('n', (8, 1)),
///      Branch
///        (('k', (6, 2)),
///         Branch
///           (('c', (2, 3)),Branch (('a', (1, 4)),Empty,Empty),
///            Branch
///              (('h', (5, 4)),
///               Branch (('g', (4, 5)),Branch (('e', (3, 6)),Empty,Empty),Empty),
///               Empty)),Branch (('m', (7, 3)),Empty,Empty)),
///      Branch
///        (('u', (12, 2)),
///         Branch
///           (('p', (9, 3)),Empty,
///            Branch (('s', (11, 4)),Branch (('q', (10, 5)),Empty,Empty),Empty)),
///         Empty))

(*[omit:(Solution)]*)
let layout tree =
    let next x = function
        | Empty -> x
        | Branch (_, _ , Branch ((_,(x,_)), _, _)) -> x + 1
        | Branch ((_,(x,_)), _, _) -> x + 1
    let rec loop x y tree cont =
        match tree with
            | Empty -> cont Empty
            | Branch(a, lt, rt) ->
                loop x (y+1) lt (fun lt' -> 
                    let x' = next x lt'
                    loop (x'+ 1) (y+1) rt (fun rt' -> 
                        cont <| Branch((a,(x',y)), lt', rt')))
    loop 1 1 tree id
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 65 : Layout a binary tree (2)]
/// An alternative layout method is depicted in the illustration below: 
/// 
///     1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
/// 
/// 1                                                  (n)
///                                        /                               \
/// 2                     (k)                                                          (u) 
///                  /            \                                              /
/// 3        (c)                       (m)                             (p)
///       /       \                                                         \
/// 4  (a)         (e)                                                         (q)
///               /   \
/// 5          (d)    (g)
/// 
/// Find out the rules and write the corresponding function. Hint: On a given level, the 
/// horizontal distance between neighboring nodes is constant.
///  
/// Use the same conventions as in problem P64 and test your function in an appropriate way.
///  
/// Here is the example tree from the above illustration: 
/// 
let tree65 = Branch ('n',
                Branch ('k',
                        Branch ('c',
                                Branch ('a', Empty, Empty),
                                Branch ('e',
                                        Branch ('d', Empty, Empty),
                                        Branch ('g', Empty, Empty))
                                ),
                        Branch ('m', Empty, Empty)),
                Branch ('u',
                        Branch ('p',
                                Empty,
                                Branch ('q', Empty, Empty)),
                        Empty)) 
/// Example in F#: 
/// 
/// > layout65 tree65;;
/// val it : (char * (int * int)) Tree =
///   Branch
///     (('n', (15, 1)),
///      Branch
///        (('k', (7, 2)),
///         Branch
///           (('c', (3, 3)),Branch (('a', (1, 4)),Empty,Empty),
///            Branch
///              (('e', (5, 4)),Branch (('d', (4, 5)),Empty,Empty),
///               Branch (('g', (6, 5)),Empty,Empty))),
///         Branch (('m', (11, 3)),Empty,Empty)),
///      Branch
///        (('u', (23, 2)),
///         Branch (('p', (19, 3)),Empty,Branch (('q', (21, 4)),Empty,Empty)),
///         Empty))

(*[omit:(Solution)]*)
let height tree = tree |> foldTree (fun _ lacc racc -> 1 + max lacc racc) 0

let layout65 tree =
    let separation = 
        let depth = height tree
        fun level -> (pown 2 <|  depth - level + 1) / 2
    let rec loop x y tree cont =
        match tree with
            | Empty -> cont Empty
            | Branch(a, lt, rt) ->
                let sep = separation (y+1)
                loop (x - sep) (y+1) lt (fun lt' -> 
                    loop (x + sep) (y+1) rt (fun rt' -> 
                        cont <| Branch((a,(x, y)), lt', rt')))
    loop (separation 1 - 1) 1 tree id
(*[/omit]*)
// [/snippet]

// [snippet: (***) Problem 66 : Layout a binary tree (3)]
/// Yet another layout strategy is shown in the illustration below: 
/// 
///     1  2  3  4  5  6  7  
/// 
/// 1              (n) 
///              /     \
/// 2        (k)         (u)
///         /   \       /
/// 3     (c)   (m)   (p)
///       /  \          \    
/// 4  (a)   (e)         (q)
///          /   \
/// 5     (d)    (g)
///
/// The method yields a very compact layout while maintaining a certain symmetry in 
/// every node. Find out the rules and write the corresponding Prolog predicate. Hint:
/// Consider the horizontal distance between a node and its successor nodes. How tight 
/// can you pack together two subtrees to construct the combined binary tree?
///  
/// Use the same conventions as in problem P64 and P65 and test your predicate in an 
/// appropriate way. Note: This is a difficult problem. Don't give up too early!
///  
/// Which layout do you like most? 
/// 
/// Example in F#: 
/// 
/// > layout66 tree65;;
/// val it : (char * (int * int)) Tree =
///   Branch
///     (('n', (5, 1)),
///      Branch
///        (('k', (3, 2)),
///         Branch
///           (('c', (2, 3)),Branch (('a', (1, 4)),Empty,Empty),
///            Branch
///              (('e', (3, 4)),Branch (('d', (2, 5)),Empty,Empty),
///               Branch (('g', (4, 5)),Empty,Empty))),
///         Branch (('m', (4, 3)),Empty,Empty)),
///      Branch
///        (('u', (7, 2)),
///         Branch (('p', (6, 3)),Empty,Branch (('q', (7, 4)),Empty,Empty)),Empty))

(*[omit:(Solution)]*)
let layout66 tree = 
    // This functions places the tree on a grid with the root node on (0,1)
    let rec helper gs x y tree = 
        let guards gs = 
            let children = function
                | Branch(_, l, r) -> [r; l]
                | Empty -> []
            List.collect children gs

        let isNotGuarded x = function
            | Branch((_,(x', _)), _, _)::_ -> x > x'
            | _ -> true

        let rec placeNode gs a x y radius l r =
            match helper gs (x + radius) (y + 1) r with
                | None -> placeNode gs a (x + 1) y (radius + 1) l r // increase the radius
                | Some r' -> Some <| Branch ((a,(x,y)), l, r')

        match tree with
            | Empty -> Some Empty
            | Branch(a, l, r) when isNotGuarded x gs ->
                helper (guards gs) (x - 1) (y + 1) l 
                |> Option.bind(fun l' -> placeNode (l' :: guards gs) a x y 1 l' r)
            | _ -> None

    // find the X coordinate of the farthest node to the left
    let rec findX = function
        | Branch((_,(x,_)), Empty , _) -> x 
        | Branch(_, l , _) -> findX l
        | Empty -> 0

    let tree' = helper [] 0 1 tree |> Option.get
    let minX = -1 + findX tree'

    // translate the tree so that the farthest node to the left is on the 1st column.
    foldTree (fun (a,(x,y)) lacc racc -> Branch((a,(x-minX,y)), lacc, racc) ) Empty tree'
(*[/omit]*)
// [/snippet]


// [snippet: (**) Problem 67 : A string representation of binary trees]
/// Somebody represents binary trees as strings of the following type:
/// 
/// a(b(d,e),c(,f(g,))) 
///
/// a) Write a Prolog predicate which generates this string representation, if the tree is 
/// given as usual (as nil or t(X,L,R) term). Then write a predicate which does this 
/// inverse; i.e. given the string representation, construct the tree in the usual form. 
/// Finally, combine the two predicates in a single predicate tree_string/2 which can be 
/// used in both directions.
///  
/// Example in Prolog 
/// ?- tree_to_string(t(x,t(y,nil,nil),t(a,nil,t(b,nil,nil))),S).
/// S = 'x(y,a(,b))'
/// ?- string_to_tree('x(y,a(,b))',T).
/// T = t(x, t(y, nil, nil), t(a, nil, t(b, nil, nil)))
///  
/// Example in F#: 
/// 
/// > stringToTree "x(y,a(,b))";;
/// val it : string Tree =
///   Branch
///     ("x",Branch ("y",Empty,Empty),Branch ("a",Empty,Branch ("b",Empty,Empty)))
/// > "a(b(d,e),c(,f(g,)))" |> stringToTree |> treeToString = "a(b(d,e),c(,f(g,)))";;
/// val it : bool = true

(*[omit:(Solution 1)]*)
let treeToString tree = 
    let rec loop t cont =
        match t with
            | Empty -> cont ""
            | Branch(x, Empty, Empty) -> cont <| x.ToString()
            | Branch(x, lt, rt) ->
                loop lt <| fun lstr -> loop rt <| fun rstr -> cont <| x.ToString() + "(" + lstr + "," + rstr + ")"
    loop tree id
(*[/omit]*)

(*[omit:(Solution 2)]*)
// using foldTree
let treeToString' tree = tree |> foldTree (fun x lstr rstr -> if lstr = "" && rstr = "" then x.ToString() else x.ToString() + "(" + lstr + "," + rstr + ")") ""

let stringToTree str = 
    let chars = str |> List.ofSeq
    let getNodeValue xs =
        let rec loop (acc : System.Text.StringBuilder) = function
            | [] -> (acc.ToString(), [])
            | (','::xs) as rest -> acc.ToString(), rest
            | ('('::xs) as rest -> acc.ToString(), rest
            | (')'::xs) as rest-> acc.ToString(), rest
            | x::xs -> loop (acc.Append(x)) xs
        loop (new System.Text.StringBuilder()) xs
    let leaf a = Branch(a, Empty, Empty)
    let rec loop chars cont = 
        match chars with
            | [] -> cont (Empty, [])
            | (x::_) as xs -> 
                let value, rest = getNodeValue xs
                match rest with
                    | '('::','::rs -> if value = "" then cont (Empty, rs) else loop rs <| fun (rt,rs) -> cont (Branch(value, Empty, rt),rs)
                    | '('::rs -> loop rs <| fun (lt,rs) -> loop rs <| fun (rt,rs) -> cont (Branch(value, lt, rt), rs)
                    | ','::rs -> if value = "" then loop rs cont else cont (leaf value, rs)
                    | _::rs -> cont  <| if value = "" then Empty, rs else leaf value ,rs
                    | [] -> cont <| (leaf value, [])
    loop chars fst
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 68 : Preorder and inorder sequences of binary trees]
/// Preorder and inorder sequences of binary trees. We consider binary trees with 
/// nodes that are identified by single lower-case letters, as in the example of problem 
/// P67.
///  
/// a) Write predicates preorder/2 and inorder/2 that construct the preorder and inorder 
/// sequence of a given binary tree, respectively. The results should be atoms, e.g. 
/// 'abdecfg' for the preorder sequence of the example in problem P67.
///  
/// b) Can you use preorder/2 from problem part a) in the reverse direction; i.e. given a 
/// preorder sequence, construct a corresponding tree? If not, make the necessary 
/// arrangements.
///  
/// c) If both the preorder sequence and the inorder sequence of the nodes of a binary 
/// tree are given, then the tree is determined unambiguously. Write a predicate 
/// pre_in_tree/3 that does the job.
///  
/// Example in F#: 
/// 
/// Main> let { Just t = stringToTree "a(b(d,e),c(,f(g,)))" ;
///             po = treeToPreorder t ;
///             io = treeToInorder t } in preInTree po io >>= print
/// Branch 'a' (Branch 'b' (Branch 'd' Empty Empty) (Branch 'e' Empty Empty)) 

(*[omit:(Solution)]*)
let inOrder tree = 
    let rec loop tree cont =
        match tree with
            | Empty -> cont ""
            | Branch(x, lt, rt) ->
                loop lt <| fun l -> loop rt <| fun r -> cont <| l + x.ToString() + r

    loop tree id

let preOrder tree = 
    let rec loop tree cont =
        match tree with
            | Empty -> cont ""
            | Branch(x, lt, rt) ->
                loop lt <| fun l -> loop rt <| fun r -> cont <| x.ToString() + l + r

    loop tree id

// using foldTree
let inOrder' t   = foldTree (fun x l r acc -> l (x.ToString() + (r acc))) id t ""
let preOrder' t  = foldTree (fun x l r acc -> x.ToString() + l (r acc))   id t ""

let stringToTree' preO inO = 
    let split (str : string) char = let arr = str.Split([|char|]) in if arr.Length = 1 then "","" else arr.[0], arr.[1]
    let leaf x = Branch(x, Empty, Empty)
    let rec loop xss cont =
        match xss with
            | [], _ -> cont (Empty, [])
            | x::xs, inO -> 
                match split inO x with
                    | "", "" -> cont ((leaf x), xs)
                    | inOl,  "" -> loop (xs,inOl) <| fun (l, xs) -> cont (Branch(x, l, Empty), xs)
                    | "", inOr -> loop (xs, inOr) <| fun (r, xs) -> cont (Branch(x, Empty, r), xs)
                    | inOl, inOr -> loop (xs,inOl) <| fun (l, xs) -> loop (xs, inOr) <| fun (r,xs) -> cont (Branch(x, l, r), xs)
    loop ((preO |> List.ofSeq), inO) fst
(*[/omit]*)
// [/snippet]
                

// [snippet: (**) Problem 69 : Dotstring representation of binary trees.]
/// We consider again binary trees with nodes that are identified by single lower-case 
/// letters, as in the example of problem P67. Such a tree can be represented by the 
/// preorder sequence of its nodes in which dots (.) are inserted where an empty 
/// subtree (nil) is encountered during the tree traversal. For example, the tree shown in 
/// problem P67 is represented as 'abd..e..c.fg...'. First, try to establish a syntax (BNF or 
/// syntax diagrams) and then write a predicate tree_dotstring/2 which does the 
/// conversion in both directions. Use difference lists.
///  
/// Example in F#: 
/// 
/// > dotString2Tree  "abd..e..c.fg...";;
/// val it : char Tree =
///   Branch
///     ('a',Branch ('b',Branch ('d',Empty,Empty),Branch ('e',Empty,Empty)),
///      Branch ('c',Empty,Branch ('f',Branch ('g',Empty,Empty),Empty)))
/// 
/// > tree2Dotstring it;;
/// val it : string = "abd..e..c.fg..." 

(*[omit:(Solution)]*)
// using foldTree
let tree2DotString t  = foldTree (fun x l r acc -> x.ToString() + l (r acc)) (fun acc -> "." + acc) t ""

let dotString2Tree str = 
    let chars = str |> List.ofSeq
    let rec loop chars cont =
        match chars with
            | [] -> failwith "the string is not well formed"
            | '.'::xs -> cont (Empty, xs)
            | x::xs -> loop xs <| fun (l,xs) -> loop xs <| fun (r,xs) -> cont (Branch(x, l , r), xs)
    loop chars fst
(*[/omit]*)
// [/snippet]
// [snippet: Ninety-Nine F# Problems - Problems 70 - 73 - Multiway Trees]
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
///
/// 
/// A multiway tree is composed of a root element and a (possibly empty) set of successors which
/// are multiway trees themselves. A multiway tree is never empty. The set of successor trees is
/// sometimes called a forest.
///  
///                              (a)
///                            /  |  \
///                          (f) (c) (b)
///                           |      /  \
///                          (g)   (d)  (e)
/// 
/// Problem 70B 
/// 
/// (*) Check whether a given term represents a multiway tree. 
/// 
/// In Prolog or Lisp, one writes a predicate to check this. 
/// 
/// Example in Prolog: 
/// ?- istree(t(a,[t(f,[t(g,[])]),t(c,[]),t(b,[t(d,[]),t(e,[])])])).
/// Yes
///  
/// In F#, we define multiway trees as a type.
/// 
type 'a Tree = Node of 'a  * 'a Tree list
///         
/// Some example trees: 
/// 
let tree1 = Node ('a', [])

let tree2 = Node ('a', [Node ('b', [])])

let tree3 = Node ('a', [Node ('b', [Node ('c', [])])])

let tree4 = Node ('b', [Node ('d', []); Node ('e', [])])

let tree5 = Node ('a', [
                        Node ('f', [Node ('g', [])]);
                        Node ('c', []);
                        Node ('b', [Node ('d', []); Node ('e', [])])
                       ] )

/// The last is the tree illustrated above. 
/// 
/// 
/// (*) Problem 70B : Check whether a given term represents a multiway tree
/// As in problem 54A, all members of this type are multiway trees; there is no use for a 
/// predicate to test them.
/// 
// [/snippet]

// [snippet: (*) Problem 70C : Count the nodes of a multiway tree.]
/// Example in F#: 
/// 
/// > nnodes tree2;;
/// val it : int = 2 

(*[omit:(Solution)]*)
let rec nnodes = function
    | Node (_, []) -> 1
    | Node (_, xs) -> 
        let t = xs |> List.sumBy (nnodes)
        1 + t
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 70 : Tree construction from a node string.]
/// We suppose that the nodes of a multiway tree contain single characters. In the depth-first 
/// order sequence of its nodes, a special character ^ has been inserted whenever, during the 
/// tree traversal, the move is a backtrack to the previous level.
///  
/// By this rule, the tree below (tree5) is represented as: afg^^c^bd^e^^^ 
/// 
/// 
/// 
/// Define the syntax of the string and write a predicate tree(String,Tree) to construct the 
/// Tree when the String is given. Make your predicate work in both directions.
///
/// Example in F#: 
/// 
/// > string2Tree "afg^^c^bd^e^^^";;
/// val it : char Tree =
///   Node
///     ('a',
///      [Node ('f',[Node ('g',[])]); Node ('c',[]);
///       Node ('b',[Node ('d',[]); Node ('e',[])])])
/// > string2Tree "afg^^c^bd^e^^^" = tree5;;
/// val it : bool = true

(*[omit:(Solution)]*)
let string2Tree str =
    let chars = str |> List.ofSeq
    let rec loop chars stack =
        match chars with
            | '^'::xs ->
                match stack with
                    | [x] -> x
                    | tx::Node(y, ty)::stack' -> loop xs (Node(y, ty @ [tx])::stack')
                    | [] -> failwith "malformed text"
            | x::xs -> loop xs (Node(x,[])::stack)
            | [] -> failwith "malformed text"
    loop chars [] 

let tree2String tree =
    let rec loop tree =
        match tree with
            | Node(a, []) -> a.ToString() +  "^"
            | Node(a, xs) -> a.ToString() + (xs |> List.fold(fun acc x -> acc + loop x) "")  + "^"
    loop tree
(*[/omit]*)
// [/snippet]

// [snippet: (*) Problem 71 : Determine the internal path length of a tree.]
/// We define the internal path length of a multiway tree as the total sum of the path lengths
/// from the root to all nodes of the tree. By this definition, tree5 has an internal path 
/// length of 9.
///  
/// Example in F#: 
/// 
/// > ipl tree5;;
/// val it : int = 9
/// > ipl tree4;;
/// val it : int = 2

(*[omit:(Solution)]*)
let rec ipl tree = 
    let rec loop depth = function
        | Node(a, []) -> depth
        | Node(a, xs) -> depth + (xs |> List.sumBy( fun x -> loop (depth+1) x))
    loop 0 tree 
(*[/omit]*)
// [/snippet]
    
// [snippet: (*) Problem 72 : Construct the bottom-up order sequence of the tree nodes.]
/// Write a predicate bottom_up(Tree,Seq) which constructs the bottom-up sequence of the nodes
/// of the multiway tree Tree.
///  
/// Example in F#: 
/// 
/// > bottom_up tree5;;
/// val it : string = "gfcdeba"
/// > bottom_up tree4;;
/// val it : string = "deb"

(*[omit:(Solution)]*)
let bottom_up tree = 
    let rec loop = function
        | Node(a, []) -> a.ToString()
        | Node(a, xs) -> (xs |> List.fold( fun acc x -> acc + (loop x) ) "") + a.ToString()
    loop tree 
(*[/omit]*)
// [/snippet]
 
// [snippet: (**) Problem 73 : Lisp-like tree representation.]
/// There is a particular notation for multiway trees in Lisp. Lisp is a prominent 
/// functional programming language, which is used primarily for artificial intelligence 
/// problems. As such it is one of the main competitors of Prolog. In Lisp almost everything
/// is a list, just as in Prolog everything is a term.
///  
/// The following pictures show how multiway tree structures are represented in Lisp.
///  
///    (a)        (a)        (a)        (b)            (a)
///                |          |        /   \         /  |  \
///               (b)        (b)     (d)   (e)     (f)  (c)  (b)
///                           |                     |       /   \
///                          (c)                   (g)    (d)   (e)
///
///     a        (a b)    (a (b c))   (b d e)    (a (f g) c (b d e))
///
/// Note that in the "lispy" notation a node with successors (children) in the tree is always
/// the first element in a list, followed by its children. The "lispy" representation of a 
/// multiway tree is a sequence of atoms and parentheses '(' and ')', which we shall 
/// collectively call "tokens". We can represent this sequence of tokens as a Prolog list; 
/// e.g. the lispy expression (a (b c)) could be represented as the Prolog list 
/// ['(', a, '(', b, c, ')', ')']. Write a predicate tree_ltl(T,LTL) which constructs the
/// "lispy token list" LTL if the tree is given as term T in the usual Prolog notation.
///  
/// (The Prolog example given is incorrect.) 
/// 
/// Example in F#: 
/// 
/// > treeltl "(x (a (f g) c (b d e)))";;
/// val it : char list =
///   ['('; 'x'; '('; 'a'; '('; 'f'; 'g'; ')'; 'c'; '('; 'b'; 'd'; 'e'; ')'; ')';
///    ')']
///
/// > displayList tree1;;
/// val it : string = "a"
/// > displayLisp tree2;;
/// val it : string = "(a b)"
/// > displayLisp tree3;;
/// val it : string = "(a (b c))"
/// > displayLisp tree4;;
/// val it : string = "(b d e)"
/// > displayLisp tree5;;
/// val it : string = "(a (f g) c (b d e))"
///
/// > lisp2Tree "(a (f g) c (b d e))" = tree5;;
/// val it : bool = true
///
/// As a second, even more interesting exercise try to rewrite tree_ltl/2 in a way that the
/// inverse conversion is also possible.

(*[omit:(Solution)]*)
let treeltl str = str |> List.ofSeq |> List.filter((<>) ' ')

let displayLisp tree = 
    let rec loop = function
        | Node(a, []) -> a.ToString()
        | Node(a, xs) -> "(" + a.ToString() + (xs |> List.fold( fun acc x -> acc + " " + (loop x) ) "") + ")"
    loop tree 

let lisp2Tree str = 
    let tokens = treeltl str
    let rec loop tokens stack =
        match tokens with
            | ')'::xs ->
                match stack with
                    | [x] -> x
                    | tx::Node(y, ty)::stack' -> loop xs (Node(y, ty @ [tx])::stack')
                    | [] -> failwith "malformed text"
            | '('::x::xs -> loop xs (Node(x,[])::stack)
            | x::xs -> 
                match stack with
                    | [] -> loop xs [Node(x,[])]
                    | Node(y,t)::stack -> loop xs (Node(y,t @  [Node(x,[])])::stack)
            | [] -> stack |> List.head
    loop tokens []
(*[/omit]*)
// [/snippet]// [snippet: Ninety-Nine F# Problems - Problems 80 - 89 - Graphs]
/// Ninety-Nine F# Problems - Problems 80 - 89
///
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
///

// The solutions to the problems below use there definitions for Grahps
type 'a Edge = 'a * 'a

type 'a Graph = 'a list * 'a Edge list

let g = (['b';'c';'d';'f';'g';'h';'k'],[('b','c');('b','f');('c','f');('f','k');('g','h')])

type 'a Node = 'a * 'a list

type 'a AdjacencyGraph = 'a Node list

let ga = [('b',['c'; 'f']); ('c',['b'; 'f']); ('d',[]); ('f',['b'; 'c'; 'k']); 
                                                    ('g',['h']); ('h',['g']); ('k',['f'])]

// [/snippet]

// [snippet: (***) Problem 80 : Conversions]
/// Write predicates to convert between the different graph representations. With these 
/// predicates, all representations are equivalent; i.e. for the following problems you 
/// can always pick freely the most convenient form. The reason this problem is rated 
/// (***) is not because it's particularly difficult, but because it's a lot of work to 
/// deal with all the special cases. 
/// 
/// Example in F#:
/// 
/// > let g = (['b';'c';'d';'f';'g';'h';'k'],[('b','c');('b','f');
///                                                 ('c','f');('f','k');('g','h')]);;
/// 
/// > graph2AdjacencyGraph g;;
/// val it : char AdjacencyGraph =
///   [('b', ['f'; 'c']); ('c', ['f'; 'b']); ('d', []); ('f', ['k'; 'c'; 'b']);
///    ('g', ['h']); ('h', ['g']); ('k', ['f'])]
///
/// > let ga = [('b',['c'; 'f']); ('c',['b'; 'f']); ('d',[]); ('f',['b'; 'c'; 'k']); 
///                                             ('g',['h']); ('h',['g']); ('k',['f'])];;
/// 
/// > adjacencyGraph2Graph ga;;
/// val it : char Graph =
///   (['b'; 'c'; 'd'; 'f'; 'g'; 'h'; 'k'],
///    [('b', 'c'); ('b', 'f'); ('c', 'f'); ('f', 'k'); ('g', 'h')])

(*[omit:(Solution)]*)

let graph2AdjacencyGraph ((ns, es) : 'a Graph) : 'a AdjacencyGraph = 
    let nodeMap = ns |> List.map(fun n -> n, []) |> Map.ofList
    (nodeMap,es) 
    ||> List.fold(fun map (a,b) -> map |> Map.add a (b::map.[a]) |> Map.add b (a::map.[b]))
    |> Map.toList
    |> List.map(fun (a,b) -> a, b |> List.sort)
    
let adjacencyGraph2Graph (ns : 'a AdjacencyGraph) : 'a Graph= 
    let sort ((a,b) as e) = if a > b then (b, a) else e
    let nodes = ns |> List.map fst
    let edges = (Set.empty, ns) 
                ||> List.fold(fun set (a,ns) -> (set, ns) ||> List.fold(fun s b -> s |> Set.add (sort (a,b))) ) 
                |> Set.toSeq 
                |> Seq.sort 
                |> Seq.toList
    (nodes, edges)

(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 81: Path from one node to another one]
/// Write a function that, given two nodes a and b in a graph, returns all 
/// the acyclic paths from a to b.
/// 
/// Example:
/// 
/// Example in F#:
/// 
/// > paths 1 4 [(1,[2;3]);(2,[3]);(3,[4]);(4,[2]);(5,[6]);(6,[5])];;
/// val it : int list list = [[1; 2; 3; 4]; [1; 3; 4]]
///
/// > paths 2 6 [(1,[2;3]);(2,[3]);(3,[4]);(4,[2]);(5,[6]);(6,[5])];;
/// val it : int list list = []

(*[omit:(Solution)]*)

let paths start finish (g : 'a AdjacencyGraph) = 
    let map = g |> Map.ofList
    let rec loop route visited = [
        let current = List.head route
        if current = finish then
            yield List.rev route
        else
            for next in map.[current] do
                if visited |> Set.contains next |> not then
                    yield! loop (next::route) (Set.add next visited) 
    ]
    loop [start] <| Set.singleton start
(*[/omit]*)
// [/snippet]


// [snippet: (*) Problem 82: Cycle from a given node]
/// Write a predicate cycle(G,A,P) to find a closed path (cycle) P starting at a given node
///  A in the graph G. The predicate should return all cycles via backtracking.
/// 
/// Example:
/// 
/// <example in lisp>
/// Example in F#:
/// 
/// > cycle 2 [(1,[2;3]);(2,[3]);(3,[4]);(4,[2]);(5,[6]);(6,[5])];;
/// val it : int list list = [[2; 3; 4; 2]]
///
/// > cycle 1 [(1,[2;3]);(2,[3]);(3,[4]);(4,[2]);(5,[6]);(6,[5])];;
/// val it : int list list = []

(*[omit:(Solution)]*)
let cycle start (g: 'a AdjacencyGraph) = 
    let map = g |> Map.ofList
    let rec loop route visited = [
        let current = List.head route
        for next in map.[current] do
            if next = start then
                yield List.rev <| next::route
            if visited |> Set.contains next |> not then
                yield! loop (next::route) (Set.add next visited) 
    ]
    loop [start] <| Set.singleton start
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 83: Construct all spanning trees]
/// Write a predicate s_tree(Graph,Tree) to construct (by backtracking) all spanning trees 
/// of a given graph. With this predicate, find out how many spanning trees there are for 
/// the graph depicted to the left. The data of this example graph can be found in the file
/// p83.dat. When you have a correct solution for the s_tree/2 predicate, use it to define 
/// two other useful predicates: is_tree(Graph) and is_connected(Graph). Both are 
/// five-minutes tasks!
/// 
/// Example:
/// 
/// <example in lisp>
/// Example in F#:

(*[omit:(Solution needed)]*)
let solution83 = "your solution here!!"
// this is not a solution. This is still a work in progress.
type Color = White = 0 | Gray = 1 | Black = 2

let s_tree (g : 'a AdjacencyGraph) = 
    let gMap = g |> Map.ofList
    let vertices = g |> List.map fst 
    let rec loop ((vertices,edges) as g) u visited = [
        match gMap.[u] |> List.filter(fun v -> Set.contains v visited |> not) with
            | [] -> yield g
            | nodes ->
                for v in nodes do
                    for (vs,es) in loop (vertices, edges) v (Set.add v visited) do
                        yield (v::vs,(u,v)::es)
    ]
    vertices |> List.collect(fun u -> loop ([u],[]) u (Set.singleton u)) 

let gs = [(1,[2;3;4]);(2,[1;3]);(3,[1;2;4]);(4,[1;3])]

s_tree gs |> printfn "%A"
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 84: Construct the minimal spanning tree]
/// Write a predicate ms_tree(Graph,Tree,Sum) to construct the minimal spanning tree of a given
/// labelled graph. Hint: Use the algorithm of Prim. A small modification of the solution of 
/// P83 does the trick. The data of the example graph to the right can be found in the file p84.dat.
/// 
/// Example:
/// 
/// <example in lisp>
/// 
/// Example in F#: 
/// > let graphW = [('a',['b'; 'd';]); ('b',['a';'c';'d';'e';]); ('c',['b';'e';]); 
///                 ('d',['a';'b';'e';'f';]); ('e',['b';'c';'d';'f';'g';]); ('f',['d';'e';'g';]); 
///                 ('g',['e';'f';]); ];;
/// > let gwF = 
///     let weigthMap = 
///         Map [(('a','b'), 7);(('a','d'), 5);(('b','a'), 7);(('b','c'), 8);(('b','d'), 9);
///              (('b','e'), 7);(('c','b'), 8);(('c','e'), 5);(('d','a'), 5);(('d','b'), 9);
///              (('d','e'), 15);(('d','f'), 6);(('e','b'), 7);(('e','c'), 5);(('e','d'), 15);
///              (('e','f'), 8);(('e','g'), 9);(('f','d'), 6);(('f','e'), 8);(('f','g'), 11);
///              (('g','e'), 9);(('g','f'), 11);]
///     fun (a,b) -> weigthMap.[(a,b)];;
/// 
/// val graphW : (char * char list) list =
///   [('a', ['b'; 'd']); ('b', ['a'; 'c'; 'd'; 'e']); ('c', ['b'; 'e']);
///    ('d', ['a'; 'b'; 'e'; 'f']); ('e', ['b'; 'c'; 'd'; 'f'; 'g']);
///    ('f', ['d'; 'e'; 'g']); ('g', ['e'; 'f'])]
/// val gwF : (char * char -> int)
/// 
/// > prim gw gwF;;
/// val it : char Graph =
///   (['a'; 'd'; 'f'; 'b'; 'e'; 'c'; 'g'],
///    [('a', 'd'); ('d', 'f'); ('a', 'b'); ('b', 'e'); ('e', 'c'); ('e', 'g')])
/// 

(*[omit:(Solution)]*)
let prim (s : 'a AdjacencyGraph) (weightFunction: ('a Edge -> int)) : 'a Graph = 
    let map = s |> List.map (fun (n,ln) -> n, ln |> List.map(fun m -> ((n,m),weightFunction (n,m)))) |> Map.ofList
    let nodes = s |> List.map fst
    let emptyGraph = ([],[])

    let rec dfs nodes (ns,es) current visited = 
        if nodes |> Set.isEmpty then
            (List.rev ns, List.rev es)
        else
                let (a,b) as edge = ns 
                                    |> List.collect(fun n -> map.[n] 
                                                             |> List.filter(fun ((n,m),w) -> Set.contains m visited |> not) ) 
                                    |> List.minBy snd |> fst
                let nodes' = nodes |> Set.remove b
                dfs nodes' (b::ns,edge::es) b (Set.add b visited)
    match nodes with
        | [] -> emptyGraph
        | n::ns -> dfs (Set ns) ([n],[]) n (Set.singleton n) 
    
(*[/omit]*)
// [/snippet]


// [snippet: (**) Problem 85: Graph isomorphism]
/// Two graphs G1(N1,E1) and G2(N2,E2) are isomorphic if there is a bijection f: N1 -> N2 such
/// that for any nodes X,Y of N1, X and Y are adjacent if and only if f(X) and f(Y) are adjacent.
/// 
/// Write a predicate that determines whether two graphs are isomorphic. Hint: Use an open-ended
/// list to represent the function f.
/// 
/// Example:
/// 
/// <example in lisp>
/// 
/// Example in F#: 

(*[omit:(Solution needed)]*)
let solution85 = "your solution here!!"
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 86: Node degree and graph coloration]
/// a) Write a predicate degree(Graph,Node,Deg) that determines the degree of a given node.
/// 
/// b) Write a predicate that generates a list of all nodes of a graph sorted according to 
///    decreasing degree.
/// 
/// c) Use Welch-Powell's algorithm to paint the nodes of a graph in such a way that adjacent 
///    nodes have different colors.
/// 
/// 
/// Example:
/// 
/// <example in lisp>
///
/// Example in F#: 
/// > let graph = [('a',[]);('b',['c']);('c',['b';'d';'g']);('d',['c';'e']);('e',['d';'e';'f';'g']);('f',['e';'g']);('g',['c';'e';'f'])];;
/// > degree graph 'e';;
/// val it : int = 5
/// > sortByDegree graph;;
/// val it : char Node list =
///   [ ('e',['d'; 'e'; 'f'; 'g']);  ('g',['c'; 'e'; 'f']);
///     ('c',['b'; 'd'; 'g']);  ('f',['e'; 'g']);  ('d',['c'; 'e']);
///     ('b',['c']);  ('a',[])]
/// val it : int = 5
/// > colorGraph graph;;
/// val it : (char * int) list =
///   [('a', 0); ('b', 1); ('c', 0); ('d', 1); ('e', 0); ('f', 2); ('g', 1)]

(*[omit:(Solution)]*)

let degree (g: 'a AdjacencyGraph) node = 
    let es = g |> List.find(fst >> (=) node) |> snd
    // The degree of a node is the number of edges that go to the node. 
    // Loops get counted twice.
    es |> List.sumBy(fun n -> if n = node then 2 else 1)

let sortByDegreeDesc (g : 'a AdjacencyGraph) = 
    // let use this degree function instead of the one above
    // since we alredy have all the info we need right here.
    let degree (u,adj) = adj |> List.sumBy(fun v -> if v = u then 2 else 1)
    g |> List.sortBy(degree) |> List.rev

let colorGraph g = 
    let nodes = sortByDegreeDesc g
    let findColor usedColors = 
        let colors = Seq.initInfinite id
        colors |> Seq.find(fun c -> Set.contains c usedColors |> not)
    let rec greedy colorMap nodes =
        match nodes with
            | [] -> colorMap |> Map.toList
            | (n,ns)::nodes -> 
                let usedColors = ns |> List.filter(fun n -> Map.containsKey n colorMap) |> List.map(fun n -> Map.find n colorMap ) |> Set.ofList
                let color = findColor usedColors
                greedy (Map.add n color colorMap) nodes
                
    greedy Map.empty nodes

(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 87: Depth-first order graph traversal (alternative solution)]
/// Write a predicate that generates a depth-first order graph traversal sequence. The starting 
/// point should be specified, and the output should be a list of nodes that are reachable from 
/// this starting point (in depth-first order).
/// 
/// Example:
/// 
/// <example in lisp>
/// 
/// Example in F#: 
///
/// > let gdfo = (['a';'b';'c';'d';'e';'f';'g';], 
///               [('a','b');('a','c');('a','e');('b','d');('b','f');('c','g');('e','f');]) 
///               |> Graph2AdjacencyGraph;;
/// 
/// val gdfo : char AdjacencyGraph =
///   [('a', ['e'; 'c'; 'b']); ('b', ['f'; 'd'; 'a']); ('c', ['g'; 'a']);
///    ('d', ['b']); ('e', ['f'; 'a']); ('f', ['e'; 'b']); ('g', ['c'])]
/// 
/// > depthFirstOrder gdfo 'a';;
/// val it : char list = ['a'; 'e'; 'f'; 'b'; 'd'; 'c'; 'g']

(*[omit:(Solution)]*)

// The enum Color is defined on problem 83
// The algorithm comes from the book Introduction to Algorithms by Cormen, Leiserson, Rivest and Stein.
let depthFirstOrder (g : 'a AdjacencyGraph) start = 
    let nodes = g |> Map.ofList
    let color = g |> List.map(fun (v,_) -> v, Color.White) |> Map.ofList |> ref
    let pi = ref [start]

    let rec dfs u = 
        color := Map.add u Color.Gray !color
        for v in nodes.[u] do
            if (!color).[v] = Color.White then
                pi := (v::!pi)
                dfs v
        color := Map.add u Color.Black !color

    dfs start
    !pi |> List.rev

(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 88: Connected components (alternative solution)]
/// Write a predicate that splits a graph into its connected components.
/// 
/// Example:
/// 
/// <example in lisp>
/// 
/// Example in F#: 
/// > let graph = [(1,[2;3]);(2,[1;3]);(3,[1;2]);(4,[5;6]);(5,[4]);(6,[4])];;
/// > connectedComponents graph;;
/// val it : int AdjacencyGraph list =
///   [[(6, [4]); (5, [4]); (4, [5; 6])];
///    [(3, [1; 2]); (2, [1; 3]); (1, [2; 3])]]
/// > 

(*[omit:(Solution)]*)
// using problem 87 depthFirstOrder function
let connectedComponents (g : 'a AdjacencyGraph) =
    let nodes = g |> List.map fst |> Set.ofList
    let start = g |> List.head |> fst
    let rec loop acc g start nodes = 
        let dfst = depthFirstOrder g start |> Set.ofList
        let nodes' = Set.difference nodes dfst 
        if Set.isEmpty nodes' then
            g::acc
        else
            // once we have the dfst set we can remove those nodes from the graph and
            // add them to the solution and continue with the remaining nodes
            let (cg,g') = g |> List.fold(fun (xs,ys) v -> if Set.contains (fst v) dfst then (v::xs,ys) else (xs,v::ys)) ([],[])
            let start' = List.head g' |> fst
            loop (cg::acc) g' start' nodes'
    loop [] g start nodes
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 89: Bipartite graphs]
/// Write a predicate that finds out whether a given graph is bipartite.
/// 
/// Example:
/// 
/// <example in lisp>
/// 
/// Example in F#: 
///
/// > let gdfo = [('a', ['b'; 'c'; 'e']); ('b', ['a'; 'd'; 'f']); ('c', ['a'; 'g']);('d', ['b']); 
///               ('e', ['a'; 'f']); ('f', ['b'; 'e']); ('g', ['c'])];;
/// 
/// val gdfo : (char * char list) list =
///   [('a', ['b'; 'c'; 'e']); ('b', ['a'; 'd'; 'f']); ('c', ['a'; 'g']);
///    ('d', ['b']); ('e', ['a'; 'f']); ('f', ['b'; 'e']); ('g', ['c'])]
/// 
/// > isBipartite gdfo;;
/// val it : bool = true

(*[omit:(Solution)]*)
open System.Collections.Generic; // this is where Queue<'T> is defined

let isBipartite (g : 'a AdjacencyGraph) = 
    // using the breath-first search algorithm, we can compute the distances
    // from the first node to the other the nodes. If all the even distance nodes
    // point to odd nodes and viceversa, then the graph is bipartite. This works
    // for connected graphs.
    // The algorithm comes from the book Introduction to Algorithms by Cormen, Leiserson, Rivest and Stein.
    let isBipartite' (g : 'a AdjacencyGraph) = 
        let adj = g |> Map.ofList
        // The Color enum is defined on problem 83
        let mutable color = g |> List.map(fun (v,_) -> v, Color.White) |> Map.ofList
        let mutable distances = g |> List.map(fun (v,_) -> v,-1) |> Map.ofList
        let queue = new Queue<_>()
        let start = List.head g |> fst
        color <- Map.add start Color.Gray color
        distances <- Map.add start 0 distances
        queue.Enqueue(start)
        while queue.Count <> 0 do
            let u = queue.Peek()
            for v in adj.[u] do
                if color.[v] = Color.White then
                    color <- Map.add v Color.Gray color
                    distances <- Map.add v (distances.[u] + 1) distances
                    queue.Enqueue(v)
            queue.Dequeue() |> ignore
            color <- Map.add u Color.Black color
        let isEven n = n % 2 = 0
        let isOdd = isEven >> not
        let d = distances // this is just so distances can be captured in the closure below.
        g |> List.forall(fun (v,edges) -> 
                            let isOpposite = if d.[v] |> isEven then isOdd else isEven
                            edges |> List.forall(fun e -> d.[e] |> isOpposite))

    // split the graph in it's connected components (problem 88) and test each piece for bipartiteness.
    // if all the pieces are bipartite, the graph is bipartite.
    g |> connectedComponents |> List.forall isBipartite'
(*[/omit]*)
// [/snippet]
// [snippet: Ninety-Nine F# Problems - Problems 90 - 94 - Miscellaneous problems]
/// Ninety-Nine F# Problems - Problems 90 - 94
///
/// These are F# solutions of Ninety-Nine Haskell Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_Haskell_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
///
// [/snippet]

// [snippet: (**) Problem 90 : Eight queens problem]
/// This is a classical problem in computer science. The objective is to place eight queens on a 
/// chessboard so that no two queens are attacking each other; i.e., no two queens are in the 
/// same row, the same column, or on the same diagonal.
///  
/// Hint: Represent the positions of the queens as a list of numbers 1..N. Example: 
/// [4,2,7,3,6,8,5,1] means that the queen in the first column is in row 4, the queen in the 
/// second column is in row 2, etc. Use the generate-and-test paradigm.
///  
/// Example in F#: 
/// 
/// > queens 8 |> Seq.length;;
/// val it : int = 92
/// > queens 8 |> Seq.head;;
/// val it : int list = [1; 5; 8; 6; 3; 7; 2; 4]
/// > queens 20 |> Seq.head;;
/// val it : int list =
///  [1; 3; 5; 2; 4; 13; 15; 12; 18; 20; 17; 9; 16; 19; 8; 10; 7; 14; 6; 11]

(*[omit:(Solution)]*)
// instead of solving the problem for 8 queens lets solve if for N queens.
// To solve the problem we are going to start with an empty board and then we're going
// add queen to it for each row. Elimitating invalid solutions. To do that we need a function
// (invalidPosition) that detects if one queen is in conflict with another one. And another 
// function (validSolution) that would test if the queen that we're adding is not in 
// conflict with any queen already on the board. 
// Also, the solution is going to return a a sequence of solutions instead of a list.
// That way we can get one solution realy fast if that is only what we care. For example 
// getting all the solutions for a 20x20 board would take a long time, but finding 
// the first solution only takes 5 seconds.
// 

let queens n =
    let invalidPosition (x1, y1) (x2, y2) = (x1 = x2) || (y1 = y2) || abs (x1 - x2) = abs (y1 - y2)
    let validSolution (queen, board) = board |> Seq.exists (invalidPosition queen) |> not
    // With the function "loop", we're going to move one column at time, placing queens
    // on each row and creating new boards with only valid solutions.
    let rec loop boards y =
        if y = 0 then
            boards
        else
            let boards' = boards 
                       |> Seq.collect(fun board -> [1 .. n] |> Seq.map(fun x -> (x,y),board))
                       |> Seq.filter validSolution 
                       |> Seq.map(fun (pos, xs) -> pos::xs)
            loop boards' (y - 1)
    loop (Seq.singleton([])) n |> Seq.map (List.rev >> List.map fst)
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 91 : Knight's tour]
/// Another famous problem is this one: How can a knight jump on an NxN chessboard in such a way 
/// that it visits every square exactly once? A set of solutions is given on the The_Knights_Tour
/// page.
///  
/// Hints: Represent the squares by pairs of their coordinates of the form X/Y, where both X and 
/// Y are integers between 1 and N. (Note that '/' is just a convenient functor, not division!) 
/// Define the relation jump(N,X/Y,U/V) to express the fact that a knight can jump from X/Y to U/V 
/// on a NxN chessboard. And finally, represent the solution of our problem as a list of N*N knight
/// positions (the knight's tour).
///  
/// There are two variants of this problem: 
/// 1. find a tour ending at a particular square 
/// 2. find a circular tour, ending a knight's jump from the start (clearly it doesn't matter where 
///    you start, so choose (1,1))
///  
/// Example in F#: 
/// 
/// > knightsTour 8 (1,1) |> Seq.head;;
/// val it : (int * int) list =
///   [(4, 3); (6, 4); (5, 6); (4, 8); (3, 6); (5, 5); (6, 3); (4, 4); (2, 3);
///    (1, 5); (3, 4); (5, 3); (6, 5); (4, 6); (2, 7); (3, 5); (5, 4); (6, 6);
///    (4, 5); (2, 4); (1, 6); (2, 8); (4, 7); (6, 8); (8, 7); (7, 5); (8, 3);
///    (7, 1); (5, 2); (3, 1); (1, 2); (3, 3); (4, 1); (2, 2); (1, 4); (2, 6);
///    (1, 8); (3, 7); (5, 8); (7, 7); (8, 5); (7, 3); (8, 1); (6, 2); (7, 4);
///    (8, 2); (6, 1); (4, 2); (2, 1); (1, 3); (2, 5); (1, 7); (3, 8); (5, 7);
///    (7, 8); (8, 6); (6, 7); (8, 8); (7, 6); (8, 4); (7, 2); (5, 1); (3, 2);
///    (1, 1)]
///
/// > endKnightsTour 8 (4,2);;
/// val it : (int * int) list =
///   [(4, 2); (2, 1); (1, 3); (3, 2); (1, 1); (2, 3); (1, 5); (2, 7); (4, 8);
///    (6, 7); (8, 8); (7, 6); (6, 8); (8, 7); (7, 5); (8, 3); (7, 1); (5, 2);
///    (3, 1); (1, 2); (2, 4); (1, 6); (2, 8); (4, 7); (2, 6); (1, 8); (3, 7);
///    (5, 8); (7, 7); (8, 5); (7, 3); (8, 1); (6, 2); (4, 1); (2, 2); (1, 4);
///    (3, 5); (5, 6); (4, 4); (2, 5); (1, 7); (3, 8); (5, 7); (7, 8); (8, 6);
///    (7, 4); (6, 6); (4, 5); (3, 3); (5, 4); (4, 6); (6, 5); (8, 4); (7, 2);
///    (6, 4); (4, 3); (5, 1); (6, 3); (8, 2); (6, 1); (5, 3); (3, 4); (5, 5);
///    (3, 6)]
///
/// > closedKnightsTour 8;;
/// val it : (int * int) list =
///   [(2, 3); (4, 4); (6, 3); (5, 5); (4, 3); (6, 4); (5, 6); (4, 8); (3, 6);
///    (1, 5); (3, 4); (5, 3); (6, 5); (4, 6); (2, 7); (3, 5); (5, 4); (6, 6);
///    (4, 5); (2, 4); (1, 6); (2, 8); (4, 7); (6, 8); (8, 7); (7, 5); (8, 3);
///    (7, 1); (5, 2); (3, 1); (1, 2); (3, 3); (4, 1); (2, 2); (1, 4); (2, 6);
///    (1, 8); (3, 7); (5, 8); (7, 7); (8, 5); (7, 3); (8, 1); (6, 2); (7, 4);
///    (8, 2); (6, 1); (4, 2); (2, 1); (1, 3); (2, 5); (1, 7); (3, 8); (5, 7);
///    (7, 8); (8, 6); (6, 7); (8, 8); (7, 6); (8, 4); (7, 2); (5, 1); (3, 2);
///    (1, 1)]

(*[omit:(Solution)]*)
// Wikipedia has a nice article about this problem http://en.wikipedia.org/wiki/Knights_tour
//
// The way this algorithm works is like this. We create a set (board) with all the positions
// in the board that have not being used. Also we have a function (moves) that returns a 
// list of posible moves from the current position. The variable 'validMoves' is the result of
// removing all the positions returned by 'moves' that are not in the set 'board' (positions 
// that are still available). If validMoves is empty, that means that we can not move 
// anymore. If at that time the board is empty, we have a solution! Otherwise we remove the 
// current position from the board add the curent position to the tour and continue to one 
// of the valid moves.
// Now, the trick to make the algorithm converge is to move first to the valid position 
// that has the less options once we move (Warnsdorff's rule).
// 

let moves n (x,y) =
    [(x + 2, y + 1); (x + 2, y - 1); (x - 2, y + 1); (x - 2, y - 1); (x - 1, y + 2); (x - 1, y - 2); (x + 1, y + 2); (x + 1, y - 2) ] 
    |> List.filter(fun (x,y) -> x > 0 && x <= n && y > 0 && y <= n)
        
let knightsTours n start =
    let board = [1 .. n] |> List.collect(fun x -> [1 .. n] |> List.map(fun y -> (x,y))) |> Set.ofList
    let rec loop tour board = seq {
        let validMoves = tour 
                         |> List.head // the head of the tour is our current position
                         |> moves n
                         |> List.filter(fun p -> board |> Set.contains p) 
        match validMoves with
            | [] -> if board |> Set.isEmpty then yield tour // we found a solution!
            | _ -> 
                // the call to sortBy is what makes this algorithm converge fast. 
                // We want to go first to the position with the less options
                // once we move (Warnsdorff's rule).
                for p in validMoves |> List.sortBy(moves n >> List.length) do 
                    yield! loop (p::tour) <| Set.remove p board
    }
    loop [start] <| Set.remove start board

let closedKnightsTour n =
    let start = (1,1)
    let finish = moves n start |> Set.ofList
    let flip f a b = f b a
    // lets find the first solution that ends in a position next to the start
    knightsTours n start |> Seq.find(List.head >> flip Set.contains finish)

let endKnightsTour n finish =
    // just find a tour that starts with finish and reverse it!
    knightsTours n finish |> Seq.head |> List.rev
(*[/omit]*)
// [/snippet]

// [snippet: (***) Problem 92 : Von Koch's conjecture]
/// Several years ago I met a mathematician who was intrigued by a problem for which he didn't
/// know a solution. His name was Von Koch, and I don't know whether the problem has been 
/// solved since.
///  
///                                         6
///        (d)   (e)---(f)        (4)   (1)---(7)
///         |     |              1 |     | 5
///        (a)---(b)---(c)        (3)---(6)---(2)
///         |                    2 |  3     4
///        (g)                    (5)
///
/// Anyway the puzzle goes like this: Given a tree with N nodes (and hence N-1 edges). Find a 
/// way to enumerate the nodes from 1 to N and, accordingly, the edges from 1 to N-1 in such 
/// a way, that for each edge K the difference of its node numbers equals to K. The conjecture 
/// is that this is always possible.
///  
/// For small trees the problem is easy to solve by hand. However, for larger trees, and 14 is 
/// already very large, it is extremely difficult to find a solution. And remember, we don't 
/// know for sure whether there is always a solution!
///  
/// Write a predicate that calculates a numbering scheme for a given tree. What is the solution
/// for the larger tree pictured below?
///
///     (i) (g)   (d)---(k)         (p)
///        \ |     |                 |
///         (a)---(c)---(e)---(q)---(n)
///        / |     |           |
///     (h) (b)   (f)         (m)
///
/// Example in F#:  
/// > vonKoch (['d';'a';'g';'b';'c';'e';'f'],[('d', 'a');('a', 'g');('a', 'b');('b', 'e');
///                ('b', 'c');('e', 'f')]) |> Seq.head;;
///
/// val it : int list * (int * int * int) list =
///   ([4; 3; 5; 6; 2; 1; 7],
///    [(4, 3, 1); (3, 5, 2); (3, 6, 3); (6, 1, 5); (6, 2, 4); (1, 7, 6)])
///

(*[omit:(Solution)]*)
// After some searching on the internet I couldn't find an algorithm for Graceful labeling.
// So I decided to go the brute force route. I knew this would work with the first the example
// but I wasn't sure if it would work for the second tree (a tree with 14 Nodes means that we have
// 14! (87,178,291,200) posible ways to tag the tree).
// Luckly, it did!!

// To represent the trees, I decided to use a tuple with a list of nodes and a list of tuples with the edges
type 'a Graph = 'a list * ('a * 'a) list

// Here are the two examples above using that representation.
let g = (['d';'a';'g';'b';'c';'e';'f'],[('d', 'a');('a', 'g');('a', 'b');('b', 'e');('b', 'c');('e', 'f')])

let g' = (['i';'h';'g';'a';'b';'d';'c';'f';'k';'e';'q';'m';'p';'n'],[('i', 'a');('h', 'a');('a', 'b');('a', 'g');('a', 'c');('c', 'f');('c','d');('d','k');('c','e');('e','q');('q','m');('q','n');('n','p')])

// Now I knew how to generate permutations in F# from this snippet: http://fssnip.net/48
// But the problem was, that implementation was using lists and it would not work to generate the 
// 87 billion permutations for the 14 node tree. Then I remember the LazyList type in the F#
// Power Pack. Now I can generate the permutations in a lazy way and pray that a solution 
// can be found fast.
// Here is the implemetation of using LazyList.

#if INTERACTIVE 
#r "FSharp.PowerPack.dll"
#endif

open Microsoft.FSharp.Collections

// the term interleave x ys returns a  list of all possible ways of inserting 
// the element x into the list ys.
let rec interleave x = function
    | LazyList.Nil -> LazyList.ofList [ LazyList.ofList [x]]
    | LazyList.Cons(y,ys) -> LazyList.ofSeq (seq { yield LazyList.cons x (LazyList.cons y ys)
                                                   for zs in interleave x ys do
                                                       yield LazyList.cons y zs })
        
// the function perms returns a lazy list of all permutations of a list.
let rec perms = function
    | LazyList.Nil -> LazyList.ofList [LazyList.empty]
    | LazyList.Cons(x,xs) -> LazyList.concat ( LazyList.map (interleave x) (perms xs))

// Now with the problem of generating all the permutations solved. 
// It's time to tackle the real problem.
let vonKoch (nodes, edges) =
    // diff is used to compute the edge difference acording the the map m
    let diff (m : Map<_, _>) (a,b) = abs <| m.[a] - m.[b]
    let size = nodes |> List.length
    let edgSize = edges |> List.length
    match nodes with
        | [] -> failwith "Empty graph!!"
        | _  when size <> (edgSize + 1) -> // make sure that we have a valid tree
            failwith "The tree doesn't have N - 1 egdes. Where N is the number of nodes"
        | _  -> 
            seq {
            for p in perms <| LazyList.ofList [1 .. size] do
                let sol = LazyList.toList p
                let m = sol |> List.zip nodes  |> Map.ofList
                // I'm using Set here to filter out any duplicates. 
                // It's faster than Seq.distinct
                let count = edges |> List.map (diff m) |> Set.ofList |> Set.count 
                // if the number of distint differences is equal to the number 
                // of edges, we found a solution!
                if count = edgSize then 
                    yield (sol, edges |> List.map (fun ((a,b) as e) -> m.[a], m.[b], diff m e))
            }
(*[/omit]*)
// [/snippet]


// [snippet: (***) Problem 93 : An arithmetic puzzle]
/// Given a list of integer numbers, find a correct way of inserting arithmetic signs (operators)
/// such that the result is a correct equation. Example: With the list of numbers [2,3,5,7,11] we 
/// can form the equations 2-3+5+7 = 11 or 2 = (3*5+7)/11 (and ten others!).
///  
/// Division should be interpreted as operating on rationals, and division by zero should be 
/// avoided.
///  
/// Example in F#: 
/// 
/// > solutions [2;3;5;7;11] |> List.iter (printfn "%s");;
/// 2 = 3 - (5 + (7 - 11))
/// 2 = 3 - ((5 + 7) - 11)
/// 2 = (3 - 5) - (7 - 11)
/// 2 = (3 - (5 + 7)) + 11
/// 2 = ((3 - 5) - 7) + 11
/// 2 = ((3 * 5) + 7) / 11
/// 2 * (3 - 5) = 7 - 11
/// 2 - (3 - (5 + 7)) = 11
/// 2 - ((3 - 5) - 7) = 11
/// (2 - 3) + (5 + 7) = 11
/// (2 - (3 - 5)) + 7 = 11
/// ((2 - 3) + 5) + 7 = 11
/// val it : unit = ()
///

(*[omit:(Solution)]*)
// This is similar to "The countdow problem" on chapter 11 in the book
// Programming in Haskell by Graham Hutton

// First let's define our operations. The ToString override is there to help
// on printing the solutions later on.
type Op = Add | Sub | Mult | Div
    with override op.ToString() =
            match op with
            | Add -> "+"
            | Sub -> "-"
            | Mult -> "*"
            | Div -> "/"

// Here we define wich opertaions are valid.
// For Add or Sub there is no problem
// For Mult we dont want trivial mutiplications by 1. Although the 
// problem statement is not clear if that is an issue.
// For Div we don't want division by 0, by 1 or fractions
let valid op x y =
    match op with
        | Add -> true
        | Sub -> true
        | Mult -> x <> 1 && y <> 1 
        | Div  -> y <> 0 && y <> 1 && x % y = 0

// this is function applies the operation to the x and y arguments
let app op x y =
    match op with
        | Add -> x + y
        | Sub -> x - y
        | Mult -> x * y
        | Div -> x / y

// Now, we define our expresions. This is how are we going to build the
// solutions
type Expr = Val of int | App of Op * Expr * Expr

// Just for fun, I implemented the fold function for our expresions.
// There was no need since we only use it once on the toString function.
let foldExpr fval fapp expr =
    let rec loop expr cont =
        match expr with
            | Val n -> cont <| fval n
            | App(op, l, r) -> loop l <| fun ls -> loop r <| fun rs -> cont <| fapp op ls rs
    loop expr id

// Once we have fold over expresions impelmenting toString is a one-liner.
// The code after the fold is just to remove the outher parentesis.
let toString exp = 
    let str = exp |> foldExpr string (fun op l r -> "(" + l + " " + string op + " " + r + ")")
    if str.StartsWith("(") then
        str.Substring(1,str.Length - 2)
    else
        str

// The 'eval' function returns a sigleton list with the result of the evaluation.
// If the expresion is not valid, returns the empty list ([])
let rec eval = function
    | Val n -> [n]
    | App(op, l, r) -> 
        [for x in eval l do
            for y in eval r do
                if valid op x y then
                    yield app op x y]

// The function 'init', 'inits', 'tails' are here to help implement the 
// function splits and came from haskell

// the function inits accepts a list and returns the list without its last item
let rec init = function
    | [] -> failwith "empty list!"
    | [_] -> []
    | x::xs -> x :: init xs
    
// The function inits returns the list of all initial segments
// of a list , in order of increasing length.
// Example:
// > inits [1..4];;
// val it : int list list = [[]; [1]; [1; 2]; [1; 2; 3]; [1; 2; 3; 4]]
let rec inits = function
    | [] -> [[]]
    | x::xs -> [ yield []
                 for ys in inits xs do
                     yield x::ys]

// the function tails returns the list of initial segments 
// of its argument list, shortest last
// Example:
// > tails [1..4];;
// val it : int list list = [[1; 2; 3; 4]; [2; 3; 4]; [3; 4]; [4]; []]
let rec tails = function
    | [] -> [[]]
    | x::xs as ls -> 
        [ yield ls
          for ys in tails xs do
              yield ys ]

// this is what drives the solution to this problem and 
// came from the haskell solution.
// Here is an example of its use:
// > splits [1..4];;
// val it : (int list * int list) list =
//  [([1], [2; 3; 4]); ([1; 2], [3; 4]); ([1; 2; 3], [4])]
// As you can see, it returs all the ways we can split a list.
let splits xs = List.tail (init (List.zip (inits xs) (tails xs)))

// Now that we're armed with all these functions, we're ready to tackle the real problem.

// The goal of the function expressions is to build all valid expressions and its value given a 
// list  of numbers. First we split the list in all posible ways (1). Then we take
// the left side of the split and build all the valid expresions (2). We do the same for the
// right side (3). Now we combine the two expresions with all the operators (4). If the operation
// is valid, we add it to the list of expressions (5,6).
let rec expressions = function
    | [x] -> [(Val x, x)]
    | xs  -> [ for xsl, xsr in splits xs do (* 1 *)
                for (expl, vall) in expressions xsl do (* 2 *)
                    for (expr, valr) in expressions xsr do (* 3 *)
                        for op in [Add; Sub; Mult; Div] do (* 4 *)
                            if valid op vall valr then (* 5 *)
                                yield (App (op, expl, expr) ,app op vall valr) (* 6 *)]


// Now that we have a way of generating valid expressions, it's time to
// generate the equaions. Again, we split the list of numbers (1). Then we generate the 
// list of expressions from the left side of the split (2). Same with the right side (3).
// If both expressions have the same value, add it to our soutions (4,5).
let equations = function
    | []  -> failwith "error: empty list"
    | [_] -> failwith "error: singleton list"
    | xs  -> [for xsl, xsr in splits xs do (* 1 *)
                for el, vl in expressions xsl do (* 2 *)
                    for er, vr in expressions xsr do (* 3 *)
                        if vl = vr then (* 4 *)
                            yield (el, er) (* 5 *)]

// Go thought the list of equations a pretty-print them.
let solutions = equations >> List.map(fun (exp1, exp2) -> toString exp1 + " = " + toString exp2)
(*[/omit]*)
// [/snippet]

// [snippet: (***) Problem 94 : Generate K-regular simple graphs with N nodes]
/// In a K-regular graph all nodes have a degree of K; i.e. the number of edges incident in each 
/// node is K. How many (non-isomorphic!) 3-regular graphs with 6 nodes are there?

(*[omit:(Solution needed)]*)
let solution94 = "your solution here!!"
(*[/omit]*)
// [/snippet]// [snippet: Ninety-Nine F# Problems - Problems 95 - 99 - Miscellaneous problems]
///
/// These are F# solutions of Ninety-Nine F# Problems 
/// (http://www.haskell.org/haskellwiki/H-99:_Ninety-Nine_F#_Problems), 
/// which are themselves translations of Ninety-Nine Lisp Problems
/// (http://www.ic.unicamp.br/~meidanis/courses/mc336/2006s2/funcional/L-99_Ninety-Nine_Lisp_Problems.html)
/// and Ninety-Nine Prolog Problems
/// (https://sites.google.com/site/prologsite/prolog-problems).
///
/// If you would like to contribute a solution or fix any bugs, send 
/// an email to paks at kitiara dot org with the subject "99 F# problems". 
/// I'll try to update the problem as soon as possible.
///
/// The problems have different levels of difficulty. Those marked with a single asterisk (*) 
/// are easy. If you have successfully solved the preceeding problems you should be able to 
/// solve them within a few (say 15) minutes. Problems marked with two asterisks (**) are of 
/// intermediate difficulty. If you are a skilled F# programmer it shouldn't take you more than 
/// 30-90 minutes to solve them. Problems marked with three asterisks (***) are more difficult. 
/// You may need more time (i.e. a few hours or more) to find a good solution
///
/// Though the problems number from 1 to 99, there are some gaps and some additions marked with 
/// letters. There are actually only 88 problems.
///
// [/snippet]

// [snippet: (**) Problem 95 : English number words]
/// On financial documents, like cheques, numbers must sometimes be written in full words. 
/// Example: 175 must be written as one-seven-five. Write a predicate full-words/1 to print 
/// (non-negative) integer numbers in full words.
///  
/// Example in F#: 
/// 
/// > fullWords 175;;
/// val it : string = "one-seven-five"

(*[omit:(Solution)]*)
let fullWords (n: int) = 
    let words = [| "zero"; "one"; "two"; "three"; "four"; "five"; "six"; "seven"; "eight"; "nine" |]
    let digits = n.ToString() |> Seq.map (fun c -> int c - int '0') |> Seq.map (fun c -> words.[c])|> Array.ofSeq
    System.String.Join("-", digits)
(*[/omit]*)
// [/snippet]

// [snippet: (**) Problem 96 : Syntax checker]
/// In a certain programming language (Ada) identifiers are defined by the syntax diagram below.
///  
///   ---->|letter|---+-------------------------------------+------>
///                   |                                     |
///                   +--------------+------>|letter|---+---+
///                   |             / \                 |
///                   +--->| - |---+   +---->|digit |---+
///                   |                                 |
///                   +---------------------------------+
///
/// Transform the syntax diagram into a system of syntax diagrams which do not contain loops;
/// i.e. which are purely recursive. Using these modified diagrams, write a predicate 
/// identifier/1 that can check whether or not a given string is a legal identifier.
///  
/// Example in Prolog: 
/// % identifier(Str) :- Str is a legal identifier 
///  
/// Example in F#: 
/// 
/// > identifier "this-is-a-long-identifier";;
/// val it : bool = true
/// > identifier "this-ends-in-";;
/// val it : bool = false
/// > identifier "two--hyphens";;
/// val it : bool = false

(*[omit:(Solution 1)]*)
// identifier = letter((-)?(letter|digit))*
// Some people, when confronted with a problem, think "I know, I'll use regular expressions." Now they have two problems.  - Jamie Zawinski
let identifier expr = System.Text.RegularExpressions.Regex.IsMatch(expr,@"^([a-z]|[A-Z])((\-)?([0-9]|[a-z]|[A-Z]))*$")
(*[/omit]*)

(*[omit:(Solution 2)]*)
// This is the overkill solution using a parser combinator.
// For a solution using fslex and fsyacc go here: https://github.com/paks/99-FSharp-Problems/tree/master/P96
// The combinator came from here: http://v2matveev.blogspot.com/2010/05/f-parsing-simple-language.html
type 'a ParserResult = Success of 'a * char list | Failed

type 'a Parser = Parser of (char list -> 'a ParserResult)

let apply (Parser p) s = p s

let run p l = apply p (Seq.toList l)

let one v = Parser(fun cs -> Success(v,cs))
let fail() = Parser(fun _ -> Failed)

let bind p f = Parser (fun cs ->
        match apply p cs with        
        | Success(r, cs2) -> apply (f r) cs2        
        | Failed -> Failed)    

let choose f p = Parser(fun cs ->
    match cs with
        | c::cs when f c -> Success(p c, cs)
        | _ -> Failed)

let (<|>) p1 p2 = Parser(fun cs ->
    match apply p1 cs with
        | Failed -> apply p2 cs
        | result -> result)

let (<&>) p1 p2 = Parser(fun cs ->
    match apply p1 cs with
        | Success(_, cs2) -> apply p2 cs2
        | Failed -> Failed)

let letter = choose System.Char.IsLetter id    

let letterOrDigit = choose System.Char.IsLetterOrDigit id    

let hiphen = choose ((=) '-') id    

type ParseBuilder() =
    member parser.Return(v) = one v
    member parser.Bind(p, f) = bind p f
    member parser.ReturnFrom(p) = p
    member parser.Zero() = fail()

let parser = new ParseBuilder()

let rec zeroOrMany p f v0 = 
    parser {
        return! oneOrMany p f v0 <|> one v0
    }

and oneOrMany p f v0 = 
    parser {
        let! v1 = p
        return! zeroOrMany p f (f v0 v1)
    }

let hiphenLetterOrDigit = (hiphen <&> letterOrDigit) <|> letterOrDigit

let identifierP = parser {
    let! l = letter
    let sb = new System.Text.StringBuilder(l.ToString())
    let! rest = sb |> zeroOrMany hiphenLetterOrDigit (fun acc v -> acc.Append(v))
    return rest.ToString()
}

let identifier' str =
    match run identifierP str with
    | Success(_,[]) -> true //if the parser consumed all the input, then it's an identifier
    | _ -> false
(*[/omit]*)
// [/snippet]

// [snippet: (***) Problem 97 : Sudoku]
/// Sudoku puzzles go like this:
///
///       Problem statement                 Solution
///
///        .  .  4 | 8  .  . | .  1  7          9  3  4 | 8  2  5 | 6  1  7         
///                |         |                          |         |
///        6  7  . | 9  .  . | .  .  .          6  7  2 | 9  1  4 | 8  5  3
///                |         |                          |         |
///        5  .  8 | .  3  . | .  .  4          5  1  8 | 6  3  7 | 9  2  4
///        --------+---------+--------          --------+---------+--------
///        3  .  . | 7  4  . | 1  .  .          3  2  5 | 7  4  8 | 1  6  9
///                |         |                          |         |
///        .  6  9 | .  .  . | 7  8  .          4  6  9 | 1  5  3 | 7  8  2
///                |         |                          |         |
///        .  .  1 | .  6  9 | .  .  5          7  8  1 | 2  6  9 | 4  3  5
///        --------+---------+--------          --------+---------+--------
///        1  .  . | .  8  . | 3  .  6          1  9  7 | 5  8  2 | 3  4  6
///                |         |                          |         |
///        .  .  . | .  .  6 | .  9  1          8  5  3 | 4  7  6 | 2  9  1
///                |         |                          |         |
///        2  4  . | .  .  1 | 5  .  .          2  4  6 | 3  9  1 | 5  7  8
///
/// Every spot in the puzzle belongs to a (horizontal) row and a (vertical) column, as well
/// as to one single 3x3 square (which we call "square" for short). At the beginning, some 
/// of the spots carry a single-digit number between 1 and 9. The problem is to fill the 
/// missing spots with digits in such a way that every number between 1 and 9 appears exactly 
/// once in each row, in each column, and in each square.

(*[omit:(Solution)]*)
let solution97 = "https://github.com/paks/ProjectEuler/blob/master/Euler2/P96/sudoku.fs"
(*[/omit]*)
// [/snippet]

// [snippet: (***) Problem 98 : Nonograms]
/// Around 1994, a certain kind of puzzle was very popular in England. The "Sunday Telegraph" 
/// newspaper wrote: "Nonograms are puzzles from Japan and are currently published each week 
/// only in The Sunday Telegraph. Simply use your logic and skill to complete the grid and 
/// reveal a picture or diagram." As a Prolog programmer, you are in a better situation: you 
/// can have your computer do the work! Just write a little program ;-).
///
/// The puzzle goes like this: Essentially, each row and column of a rectangular bitmap is 
/// annotated with the respective lengths of its distinct strings of occupied cells. The 
/// person who solves the puzzle must complete the bitmap given only these lengths.
///
///             Problem statement:          Solution:
///             |_|_|_|_|_|_|_|_| 3         |_|X|X|X|_|_|_|_| 3           
///             |_|_|_|_|_|_|_|_| 2 1       |X|X|_|X|_|_|_|_| 2 1         
///             |_|_|_|_|_|_|_|_| 3 2       |_|X|X|X|_|_|X|X| 3 2         
///             |_|_|_|_|_|_|_|_| 2 2       |_|_|X|X|_|_|X|X| 2 2         
///             |_|_|_|_|_|_|_|_| 6         |_|_|X|X|X|X|X|X| 6           
///             |_|_|_|_|_|_|_|_| 1 5       |X|_|X|X|X|X|X|_| 1 5         
///             |_|_|_|_|_|_|_|_| 6         |X|X|X|X|X|X|_|_| 6           
///             |_|_|_|_|_|_|_|_| 1         |_|_|_|_|X|_|_|_| 1           
///             |_|_|_|_|_|_|_|_| 2         |_|_|_|X|X|_|_|_| 2           
///              1 3 1 7 5 3 4 3             1 3 1 7 5 3 4 3              
///              2 1 5 1                     2 1 5 1                      
///      
/// For the example above, the problem can be stated as the two lists [[3],[2,1],[3,2],[2,2]
/// ,[6],[1,5],[6],[1],[2]] and [[1,2],[3,1],[1,5],[7,1],[5],[3],[4],[3]] which give the 
/// "solid" lengths of the rows and columns, top-to-bottom and left-to-right, respectively. 
/// Published puzzles are larger than this example, e.g. 25 x 20, and apparently always have 
/// unique solutions.
///
/// Example in F#:
///
/// > printfn "%s" <| nonogram [[3];[2;1];[3;2];[2;2];[6];[1;5];[6];[1];[2]]
///                             [[1;2];[3;1];[1;5];[7;1];[5];[3];[4];[3]];;
/// |_|X|X|X|_|_|_|_| 3
/// |X|X|_|X|_|_|_|_| 2 1
/// |_|X|X|X|_|_|X|X| 3 2
/// |_|_|X|X|_|_|X|X| 2 2
/// |_|_|X|X|X|X|X|X| 6
/// |X|_|X|X|X|X|X|_| 1 5
/// |X|X|X|X|X|X|_|_| 6
/// |_|_|_|_|X|_|_|_| 1
/// |_|_|_|X|X|_|_|_| 2
///  1 3 1 7 5 3 4 3
///  2 1 5 1
///

(*[omit:(Solution needed)]*)
let solution98 = "your solution here!!"
(*[/omit]*)

// [/snippet]

// [snippet: (***) Problem 99 : Crossword puzzle]
/// Given an empty (or almost empty) framework of a crossword puzzle and a set of words. The 
/// problem is to place the words into the framework.
///             
///                P R O L O G     E 
///                E   N     N     M
///                R   L i N U X   A
///                L   i   F   M A C    
///                    N   S Q L   S
///                    E
///                  W E B
///
/// The particular crossword puzzle is specified in a text file which first lists the words
/// (one word per line) in an arbitrary order. Then, after an empty line, the crossword 
/// framework is defined. In this framework specification, an empty character location is 
/// represented by a dot (.). In order to make the solution easier, character locations can 
/// also contain predefined character values. The puzzle above is defined in the file 
/// p99a.dat, other examples are p99b.dat and p99d.dat. There is also an example of a puzzle 
/// (p99c.dat) which does not have a solution.
/// 
/// Words are strings (character lists) of at least two characters. A horizontal or vertical 
/// sequence of character places in the crossword puzzle framework is called a site. Our 
/// problem is to find a compatible way of placing words onto sites.
/// 
/// Hints: (1) The problem is not easy. You will need some time to thoroughly understand it. 
/// So, don't give up too early! And remember that the objective is a clean solution, not 
/// just a quick-and-dirty hack!
/// 
/// (2) Reading the data file is a tricky problem for which a solution is provided in the 
/// file p99-readfile.pl. See the predicate read_lines/2.
/// 
/// (3) For efficiency reasons it is important, at least for larger puzzles, to sort the 
/// words and the sites in a particular order. For this part of the problem, the solution 
/// of P28 may be very helpful.
/// 
/// Example in F#:
/// 
/// ALPHA
/// ARES
/// POPPY
/// 
///   .
///   .
/// .....
///   . .
///   . .
///     .
/// > solve $ readCrossword "ALPHA\nARES\nPOPPY\n\n  .  \n  .  \n.....\n  . .\n  . .\n    .\n"
///  
/// [[((3,1),'A');((3,2),'L');((3,3),'P');((3,4),'H');((3,5),'A');((1,3),'P');((2,3)
/// ,'O');((3,3),'P');((4,3),'P');((5,3),'Y');((3,5),'A');((4,5),'R');((5,5),'E');((
/// 6,5),'S')]]

(*[omit:(Solution needed)]*)
let solution99 = "your solution here!!"
(*[/omit]*)

// [/snippet]namespace Euler
open System.Numerics
open System


module Library = 
    // newton's method sqrt
    let rec square_root x guess =
        let next_guess = guess - (guess * guess - x)/(2L * guess)
        if (abs (guess - next_guess) < 1L) then 
            next_guess
        else 
            square_root x next_guess

    // prime number sieve
    let sieve max = 
        let numbers = [2L..max]
        let sqrt_max = square_root max 10L
        let rec findprimes (primeslist:list<int64>) primes = 
            if primeslist.Head > sqrt_max  then
                primes @ primeslist
            else
                let new_primes = primeslist.Head :: primes
                let new_primeslist = List.filter (fun x -> x % primeslist.Head <> 0L) primeslist
                findprimes new_primeslist new_primes
        findprimes numbers []
        
    // factorial 
    let rec factorial x =  
        match x with 
            | _ when x = 1I -> 1I
            | _ when x <= 0I -> 0I
            | _ -> x * factorial (x-1I)
    
    // greatest common divisor
    let rec gcd x y = 
      match x with 
        | _ when x > y -> gcd y (x-y)
        | _ when x < y -> gcd x (y-x)
        | _ -> x 

    // least common multiple
    let lcm x y = abs (x*y)/gcd x y 

    // map a list to an integer, then sum
    let maptoIntAndSum = Array.map(fun (x:char) -> Convert.ToInt32(x)-48) >> Array.sum

    // n choose k
    let n_choose_k n k = factorial n/(factorial k * factorial (n-k))

    // n choose k with repetitiona
    let n_choose_k_repetitions n k = factorial (n+k-1I)/(factorial k * factorial (n-1I))

// Learn more about F# at http://fsharp.net
namespace Euler
open Library

// Add all the natural numbers below one thousand that are multiples of 3 or 5.
module Prob1 = 
    let all = [1 .. 999]
    let filtered = all |> List.filter (fun x -> x%3=0 || x%5=0)
    let result = List.sum filtered

// By considering the terms in the Fibonacci sequence whose values do not exceed four million, find the sum of the even-valued terms.
module Prob2 = 
    let fibSeq = Seq.unfold (fun (a,b) -> Some( a+b, (b, a+b) ) ) (0,1)
    let fibList = fibSeq |> Seq.takeWhile (fun x -> x<4000000 ) |> Seq.toList
    let filterList = List.filter (fun x -> x%2=0) fibList
    let result = List.sum filterList

// What is the difference between the sum of the squares and the square of the sums?
module Prob6 = 
    let numbers = [1..100]   
    let square x = x*x 
    let result = (numbers |> List.sum |> square)-(numbers |> List.map square |> List.sum)

// Calculate the sum of all the primes below two million.
module Prob10 = 
    let result = sieve 2000000L |> List.sum
namespace Euler
open Library

// What is the sum of the digits of the number 2^1000?
module prob16 = 
    let result = (2I ** 1000).ToString().ToCharArray() |> maptoIntAndSum

// Find the sum of the digits in the number 100!
module prob20 = 
    let fact100 = factorial 100I
    let chararray = fact100.ToString().ToCharArray()
    let result = maptoIntAndSum chararray
// Learn more about F# at http://fsharp.net
namespace Euler
open Library

// What is the first term in the Fibonacci sequence to contain 1000 digits?
module Prob25 = 
  let fibonnacci = Seq.unfold(fun (a,b) -> Some( a+b, (b, a+b) )) (0I,1I)
  let result = (Seq.findIndex(fun elem -> elem.ToString().ToCharArray().Length >= 1000) fibonnacci) + 2
namespace Euler
open Library
open System

module Working = 

    // factor 
    let factorization x = 
      let primeslist = sieve x

      let rec factors acc (y:int64) (z:int64) = 
        match y%z with 
          | 0L when y = z-> z::acc
          | 0L when y <> z -> factors (z::acc) (y/z) z
          | _ -> acc

    // What is the largest prime factor of the number 600851475143 ?
    let prob3 x = x*x

        
    // What is the smallest number divisible by each of the numbers 1 to 20?
    let prob5 = 
        let primes10 = sieve 10L

    // Find the sum of all numbers which are equal to the sum of the factorial of their digits. prob 34
    let max = [3I..9I] |> List.map factorial |> List.sum 

    let combos = 
      seq {
        for k in [1I..9I] do
          yield n_choose_k_repetitions 9I k
      } |> Seq.sum
      
    