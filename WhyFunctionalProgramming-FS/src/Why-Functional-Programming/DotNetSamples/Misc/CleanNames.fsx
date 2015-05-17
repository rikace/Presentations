    open System
    open System.Text
    let appendComman (sb:StringBuilder) =  sb.Append(",")

    let capitalieString (s:string) = s.Substring(0, 1).ToUpper() + s.Substring(1, s.Length)

    let cleanNames (names:string list) =
        names 
        |> List.filter(fun name -> name.Length > 1)
        |> List.map (capitalieString)
        |> List.fold(fun( acc:StringBuilder) name -> let sb' = acc.Append(name) 
                                                     sb'.Append(",")) (new StringBuilder())





    // multiply :: int -> int -> int
    let multiply x1 x2 = x1 * x2

    // multiply :: int -> int -> int
    let multiply' = fun x1 -> fun x2 -> x1 * x2
    
    let value = multiply 2 3



    // double :: int -> int
    let double x1 = multiply 2 x1    
    let double' = multiply 2
    
    let value'  = double 3
    let value'' = double 5
    
    double 5
    double' 5

    // building blocks
    let add2 x = x + 2
    let mult3 x = x * 3
    let square x = x * x

    [1..10] |> List.map add2 |> printf "%A"
    [1..10] |> List.map mult3 |> printf "%A"
    [1..10] |> List.map square |> printf "%A"

    // Pipeline operator
    // 'a -> ('a -> 'b) -> 'b
    let (|>) x f = f x
    
    // Compose operator
    // (('a -> 'b) -> ('b -> 'c) -> 'a -> 'c)
    let (>>) g f x = f(g(x))
    let add2Mult3 = add2 >> mult3
    let mult3Square = mult3 >> square

    [1..10] |> List.map add2Mult3 |> printf "%A"

    let double_then_sum = 
        List.map ((*)2) >> List.reduce (+)

    double_then_sum [1..5]
    