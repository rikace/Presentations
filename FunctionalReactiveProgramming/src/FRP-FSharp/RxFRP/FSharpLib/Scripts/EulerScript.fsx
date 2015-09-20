#light

open System
open System.IO

(*
Ios with Mono touch
c++
Tablet ??

javascript + html

mac - apple name in tool bar ecc...

*)

let dir = @"E:\ProjectEuler-master"
let destFile = @"c:\temp\AggregateFS.txt"

let rec getfiles path = [ for f in Directory.GetFiles(path, "Problem*.fs") do yield f
                          for d in Directory.GetDirectories(path) do yield! getfiles d ];;

let writeToFile sourceFile destinationFile = 
    let lines = File.ReadAllLines(sourceFile)
    use writer = new FileStream(destinationFile, FileMode.Append)
    use sw = new StreamWriter(writer)
    sw.WriteLine("//=================> " + Path.GetFileNameWithoutExtension(sourceFile)  + " <=================================")
    sw.WriteLine()
    for line in lines do
        sw.WriteLine(line)
    sw.WriteLine()
    sw.Flush()

let writeToFileDirec f = writeToFile f destFile

getfiles dir
|> List.iter(fun f ->   printfn "%s" f
                        writeToFileDirec f)

//////////////////////////////////

//=================> Problem001 <=================================

let answer =  seq {for x in 1..999 do if x%5=0 || x%3=0 then yield x} |> Seq.sum
printfn "answer = %d" answer

//=================> Problem002 <=================================

let answer = 
   Seq.unfold(fun (p, c) -> Some((p, c), (c, p+c))) (1,2)
   |>Seq.map(fun (p, c) -> c)
   |>Seq.takeWhile(fun x -> x<4000000)
   |>Seq.filter(fun x -> x%2=0)
   |>Seq.sum
   
printfn "%d" answer

//=================> Problem004 <=================================

let revString (str:string) =
   new string(str.ToCharArray() |> Array.rev)

let answer =
   seq {
      for x in [100..999] do
         for y in [100..999] do
            yield x*y
   }
   |> Seq.map(fun x -> (x, x.ToString()))
   |> Seq.filter(fun (x,s) -> s = (revString s))
   |> Seq.maxBy fst
   |> fst

printfn "answer = %d" answer

//=================> Problem005 <=================================

let answer = 
   Seq.unfold(fun x -> Some(x, x+1L)) (1L)
   |> Seq.find(fun x-> x%3L=0L && x%7L=0L && x%11L=0L && x%12L=0L && x%13L=0L && x%14L=0L && x%15L=0L && x%16L=0L && x%17L=0L && x%18L=0L && x%19L=0L && x%20L =0L)

printfn "answer = %d" answer

//=================> Problem006 <=================================

let sumOfSquares = seq {for x in 1..100 -> x*x} |> Seq.sum
let squareOfSums = (([1..100] |> Seq.sum |> float) ** 2.0) |> int
let answer = squareOfSums - sumOfSquares

printfn "answer = %d" answer 

//=================> Problem007 <=================================

let answer = 
   Seq.initInfinite(fun x -> x+1)
   |> Seq.filter(fun x -> [2..x|> float |> sqrt |> int] |> Seq.forall(fun y -> y=x || x%y<>0))
   |> Seq.nth 10001

printfn "answer = %d" answer

//=================> Problem008 <=================================

let string =  System.Text.RegularExpressions.Regex.Replace(
                            "73167176531330624919225119674426574742355349194934
                             96983520312774506326239578318016984801869478851843
                             85861560789112949495459501737958331952853208805511
                             12540698747158523863050715693290963295227443043557
                             66896648950445244523161731856403098711121722383113
                             62229893423380308135336276614282806444486645238749
                             30358907296290491560440772390713810515859307960866
                             70172427121883998797908792274921901699720888093776
                             65727333001053367881220235421809751254540594752243
                             52584907711670556013604839586446706324415722155397
                             53697817977846174064955149290862569321978468622482
                             83972241375657056057490261407972968652414535100474
                             82166370484403199890008895243450658541227588666881
                             16427171479924442928230863465674813919123162824586
                             17866458359124566529476545682848912883142607690042
                             24219022671055626321111109370544217506941658960408
                             07198403850962455444362981230987879927244284909188
                             84580156166097919133875499200524063689912560717606
                             05886116467109405077541002256983155200055935729725
                             71636269561882670428252483600823257530420752963450", "\\D", System.String.Empty)

let answer = 
   string
   |> Seq.map(fun x -> int x - 0x30)
   |> Seq.windowed 5
   |> Seq.map(Seq.reduce(fun acc x -> acc * x))
   |> Seq.max


printfn "answer = %d" answer

//=================> Problem009 <=================================

let answer = 
   seq {
   for a in [1.0..500.0] do
      for b in [1.0..500.0] do
         let c = 1000.0-a-b
         yield (a,b,c)
   } |> Seq.filter(fun (a,b,c) -> a**2.0 + b**2.0 = c **2.0)
     |> Seq.map(fun (a,b,c) -> a*b*c)
     |> Seq.head
     |> int

printfn "answer = %d" answer

//=================> Problem010 <=================================

let isPrime n = [|2L.. (n|>float|>sqrt|>int64)|] |> Array.exists(fun y -> n <> y && n % y=0L) |> not

let answer = 
   seq {
         yield 2L
         yield!
            Async.Parallel [ for i in [3L..2L..2000000L] -> async { return (i, isPrime(i)) } ]
            |> Async.RunSynchronously
            |> Seq.filter snd
            |> Seq.map fst

      }
   |> Seq.sum

printfn "answer = %d" answer

//=================> Problem12 <=================================

let triangleNumbers = Seq.unfold(fun (p, c) -> Some((p, c), (p+1, c+p))) (2, 1) |> Seq.map(fun (p, c) -> c)
let countDivisors x = [1..x|>float|>sqrt|>int]
                      |> Seq.filter(fun y -> x%y=0)
                      |> Seq.map(fun y -> [y;x/y])
                      |> Seq.concat
                      |> Seq.length

let answer = 
   triangleNumbers
   |> Seq.filter(fun x -> (countDivisors x) > 500)
   |> Seq.head

printfn "answer = %d" answer

//=================> Problem013 <=================================

open System.Text.RegularExpressions

let numbersString = "37107287533902102798797998220837590246510135740250
                     46376937677490009712648124896970078050417018260538
                     74324986199524741059474233309513058123726617309629
                     91942213363574161572522430563301811072406154908250
                     23067588207539346171171980310421047513778063246676
                     89261670696623633820136378418383684178734361726757
                     28112879812849979408065481931592621691275889832738
                     44274228917432520321923589422876796487670272189318
                     47451445736001306439091167216856844588711603153276
                     70386486105843025439939619828917593665686757934951
                     62176457141856560629502157223196586755079324193331
                     64906352462741904929101432445813822663347944758178
                     92575867718337217661963751590579239728245598838407
                     58203565325359399008402633568948830189458628227828
                     80181199384826282014278194139940567587151170094390
                     35398664372827112653829987240784473053190104293586
                     86515506006295864861532075273371959191420517255829
                     71693888707715466499115593487603532921714970056938
                     54370070576826684624621495650076471787294438377604
                     53282654108756828443191190634694037855217779295145
                     36123272525000296071075082563815656710885258350721
                     45876576172410976447339110607218265236877223636045
                     17423706905851860660448207621209813287860733969412
                     81142660418086830619328460811191061556940512689692
                     51934325451728388641918047049293215058642563049483
                     62467221648435076201727918039944693004732956340691
                     15732444386908125794514089057706229429197107928209
                     55037687525678773091862540744969844508330393682126
                     18336384825330154686196124348767681297534375946515
                     80386287592878490201521685554828717201219257766954
                     78182833757993103614740356856449095527097864797581
                     16726320100436897842553539920931837441497806860984
                     48403098129077791799088218795327364475675590848030
                     87086987551392711854517078544161852424320693150332
                     59959406895756536782107074926966537676326235447210
                     69793950679652694742597709739166693763042633987085
                     41052684708299085211399427365734116182760315001271
                     65378607361501080857009149939512557028198746004375
                     35829035317434717326932123578154982629742552737307
                     94953759765105305946966067683156574377167401875275
                     88902802571733229619176668713819931811048770190271
                     25267680276078003013678680992525463401061632866526
                     36270218540497705585629946580636237993140746255962
                     24074486908231174977792365466257246923322810917141
                     91430288197103288597806669760892938638285025333403
                     34413065578016127815921815005561868836468420090470
                     23053081172816430487623791969842487255036638784583
                     11487696932154902810424020138335124462181441773470
                     63783299490636259666498587618221225225512486764533
                     67720186971698544312419572409913959008952310058822
                     95548255300263520781532296796249481641953868218774
                     76085327132285723110424803456124867697064507995236
                     37774242535411291684276865538926205024910326572967
                     23701913275725675285653248258265463092207058596522
                     29798860272258331913126375147341994889534765745501
                     18495701454879288984856827726077713721403798879715
                     38298203783031473527721580348144513491373226651381
                     34829543829199918180278916522431027392251122869539
                     40957953066405232632538044100059654939159879593635
                     29746152185502371307642255121183693803580388584903
                     41698116222072977186158236678424689157993532961922
                     62467957194401269043877107275048102390895523597457
                     23189706772547915061505504953922979530901129967519
                     86188088225875314529584099251203829009407770775672
                     11306739708304724483816533873502340845647058077308
                     82959174767140363198008187129011875491310547126581
                     97623331044818386269515456334926366572897563400500
                     42846280183517070527831839425882145521227251250327
                     55121603546981200581762165212827652751691296897789
                     32238195734329339946437501907836945765883352399886
                     75506164965184775180738168837861091527357929701337
                     62177842752192623401942399639168044983993173312731
                     32924185707147349566916674687634660915035914677504
                     99518671430235219628894890102423325116913619626622
                     73267460800591547471830798392868535206946944540724
                     76841822524674417161514036427982273348055556214818
                     97142617910342598647204516893989422179826088076852
                     87783646182799346313767754307809363333018982642090
                     10848802521674670883215120185883543223812876952786
                     71329612474782464538636993009049310363619763878039
                     62184073572399794223406235393808339651327408011116
                     66627891981488087797941876876144230030984490851411
                     60661826293682836764744779239180335110989069790714
                     85786944089552990653640447425576083659976645795096
                     66024396409905389607120198219976047599490197230297
                     64913982680032973156037120041377903785566085089252
                     16730939319872750275468906903707539413042652315011
                     94809377245048795150954100921645863754710598436791
                     78639167021187492431995700641917969777599028300699
                     15368713711936614952811305876380278410754449733078
                     40789923115535562561142322423255033685442488917353
                     44889911501440648020369068063960672322193204149535
                     41503128880339536053299340368006977710650566631954
                     81234880673210146739058568557934581403627822703280
                     82616570773948327592232845941706525094512325230608
                     22918802058777319719839450180888072429661980811197
                     77158542502016545090413245809786882778948721859617
                     72107838435069186155435662884062257473692284509516
                     20849603980134001723930671666823555245252804609722
                     53503534226472524250874054075591789781264330331690"

let numbers =
   Regex.Split(numbersString, System.Environment.NewLine)
   |> Array.map(fun x -> System.Text.RegularExpressions.Regex.Replace(x, "\\D", ""))
   |> Array.map(fun x -> bigint.Parse(x))
   |> Array.sum
   |> string

printfn "%s" (numbers.Substring(0, 10))

//=================> Problem014 <=================================

let collatz n =
   let rec loop(n,x) = 
      match n with
      | n when n = 1L -> x
      | n when n%2L = 0L -> loop(n/2L,x+1)
      | _ -> loop(3L*n+1L,x+1)
   loop(n,1)

let answer =
    Async.Parallel [ for i in 13L..999999L -> async { return (i, collatz i) } ]
    |> Async.RunSynchronously
    |> Seq.maxBy snd
printfn "answer = %d steps = %d" (fst answer) (snd answer)

//=================> Problem015 <=================================

let seq row
      = Seq.unfold (fun (entry, col) -> Some((entry, col), ((entry * (row + 1L - col)) / col, col + 1L))) (1L,1L)

let pascalTriangle = 
     seq (20L*2L)
     |> Seq.skipWhile(fun (entry,col) -> col <= 20L)
     |> Seq.head
     |> fst

printfn "answer = %d" pascalTriangle

//=================> Problem016 <=================================

let bigNumber pow =
     2I**pow
     |> string
     |> Seq.map(fun c -> int c - 0x30)
     |> Seq.sum

printfn "answer = %d" (bigNumber 1000)

//=================> Problem017 <=================================

open System.Text.RegularExpressions

let ones = [|""; "one"; "two"; "three"; "four"; "five"; "six"; "seven"; "eight"; "nine"|]
let tens = [|""; "ten"; "twenty"; "thirty"; "forty"; "fifty"; "sixty"; "seventy"; "eighty"; "ninety"|]
let teen = [|"ten"; "eleven"; "twelve"; "thirteen"; "fourteen"; "fifteen"; "sixteen"; "seventeen"; "eighteen"; "nineteen"|]

let wordify n = 
    let word = 
        let rec dev s i = 
            let str = n.ToString()
            if (i = str.Length) then s
            else
                let digit = int(str.Substring(str.Length - 1 - i, 1))
                let remainder = int(str.Substring(str.Length - 1 - i))
                let mutable build = s
                let word = 
                    match i with
                    | 0 -> ones.[digit]
                    | 1 -> 
                            match tens.[digit] with
                            | "ten" -> build<-""; teen.[int(str.Substring(str.Length - i, 1))]
                            | a -> a
                    | 2 -> ones.[digit] + " hundred and"
                    | 3 -> ones.[digit] + " thousand"
                dev (word + " " + build) (i+1)
        dev "" 0
    if n = 1000 then "one thousand" else
        let mutable trim = word.Trim()
        if trim.EndsWith(" and") then trim.Substring(0, trim.Length - 4) else trim

let count (n:string) = 
    n.Replace(" ", "").Replace(" ", "").Length

let answer = 
    [1..1000]
    |> Seq.map wordify
    |> Seq.map count
    |> Seq.sum

printfn "answer = %i" answer

//=================> Problem018 <=================================

let answer = 
   System.IO.File.ReadLines(".\\triangle.txt")
   |> Seq.map(fun x -> x.Split(' ') |> Seq.map(int))
   |> Seq.fold(fun acc x -> 
            let rowLen = Seq.length x
            x |> Seq.mapi(fun i y -> 
                              let current = fun () -> y+(Seq.nth i acc)
                              let previous = fun () -> y+(Seq.nth (i-1) acc)
                              if i=0 then current()
                              elif i=rowLen-1 then previous()
                              else max (current()) (previous())
                           )
              |> Seq.toList
         ) [0]
   |> List.max

printfn "answer = %d" answer

//=================> Problem019 <=================================

open System

let answer =
   Seq.unfold(fun (x:DateTime) -> Some(x, x.AddDays(1.0))) (DateTime(1901, 1, 1))
   |> Seq.filter(fun x -> x.DayOfWeek = DayOfWeek.Sunday && x.Day = 1)
   |> Seq.takeWhile(fun x -> x <= DateTime(2000, 12, 31))
   |> Seq.length

printfn "answer = %d" answer

//=================> Problem020 <=================================

let factorial n = [2I .. n-1I] |> Seq.reduce(fun acc n -> acc*n)
let answer = 
   (factorial 100I).ToString("R")
   |> Seq.map(fun x -> int x - 0x30)
   |> Seq.sum

printfn "answer = %d" answer

//=================> Problem021 <=================================

let d n = ([1..n|>float|>sqrt|>int]
             |> Seq.filter(fun x -> n%x=0)
             |> Seq.map(fun x -> x+n/x)
             |> Seq.sum) - n

let answer = 
   [1..9999]
   |> Seq.map(fun x -> (x, d x))
   |> Seq.filter(fun (x, y) -> x <> y && d y = x)
   |> Seq.sumBy fst

printfn "answer = %d" answer

//=================> Problem022 <=================================

let score name index = 
      (
         name
         |> Seq.map(fun c -> int(c)-64)
         |> Seq.sum
      ) *index

let answer = System.IO.File.ReadAllText(".\\names.txt").Split(',')
               |> Array.map(fun x -> x.Trim('"'))
               |> Array.sort
               |> Array.mapi(fun i x -> score x (i+1))
               |> Array.sum

printfn "answer = %d" answer

//=================> Problem023 <=================================

let divisors x =
   [1..x|>float|>sqrt|>int]
   |> Seq.filter(fun y -> x%y=0)
   |> Seq.map(fun y -> [y;x/y])
   |> Seq.concat
   |> Seq.filter(fun y -> x<>y)
   |> Seq.distinct
   |> Seq.sum


let abundantNumbers =
   [1..28123]
   |> Seq.filter(fun x -> (divisors x) > x)
   |> Seq.toList

let hasSum n = 
   abundantNumbers
   |> List.exists(fun x -> abundantNumbers |> List.exists(fun y -> n-x=y))

let answer = 
   [1..28123]
   |> Seq.filter(fun x -> not(hasSum x))
   |> Seq.sum

printfn "answer = %d" answer

//=================> Problem024 <=================================

let numbers = [0;1;2;3;4;5;6;7;8;9]

let rec permutations list taken =
    seq {
        if Set.count taken = List.length list then yield [] else
            for l in list do
                if not (Set.contains l taken) then
                    for perm in permutations list (Set.add l taken)  do
                        yield l::perm }

let p = permutations numbers Set.empty
let answer = p |> Seq.skip 999999 |> Seq.head
printfn "%A" answer

//=================> Problem025 <=================================

let fib = Seq.unfold(fun (p, c) -> Some((p, c), (c, p+c))) (1I,1I)
let answer = 
   fib
   |> Seq.mapi(fun i x -> (i+1, (fst x).ToString("R")))
   |> Seq.filter(fun (_, x) -> x.Length = 1000)
   |> Seq.head
   |> fst
printfn "answer = %d" answer

//=================> Problem026 <=================================

open System.Text
open System

let divide n = 
    let inline toInt (x:char) = int x - 0x30
    let rec solve (sb:StringBuilder, l:int, iter:int) = 
        if iter = 4000 then sb
        else 
            let result = l / n
            sb.Append(box result) |> ignore
            let multiplyBack = result * n
            let difference = l - multiplyBack
            assert (difference >= 0)
            let diffMul = difference * 10
            solve(sb, diffMul, iter+1)
    let answer = solve (new StringBuilder(), 1, 1)
    answer.ToString(1, answer.Length - 1).ToCharArray() |> List.ofArray

let compare (a:seq<_>) (b:seq<_>) =
    let equal = Seq.forall2(fun x y -> x = y)
    equal a b

let getRepGroup (chrs:List<char>) = 
    let rec windowGroup (group:List<char>) (running:List<List<char>>) = 
        match group with
        | list when list |> List.length = 1 -> windowGroup list.Tail running //One character in length. Can't repeat, so skip.
        | list -> 
            List.Empty
        | [] -> running |> List.maxBy List.length
    windowGroup chrs List.Empty

(* let answer = 
    [1..999]
    |> Seq.map(fun x -> (x, x |> divide |> countMaxRep))
    |> Seq.maxBy snd
    |> fst *)

//printfn "answer = %i" answer

let six = divide 6
printfn "%s" (new string(six |> Seq.toArray))

//=================> Problem027 <=================================

let quad n a b = n*n + a*n + b

let isPrime n = n > 0 && [|2.. (n|>float|>sqrt|>int)|] |> Array.exists(fun y -> n <> y && n % y=0) |> not

let answer = 
   seq {
   for a in {-999..999} do
      for b in {1..999} do
         let primes = 
            Seq.unfold(fun x -> Some(x, x+1)) 0
            |> Seq.takeWhile(fun x -> isPrime (quad x a b))
         yield (primes |> Seq.length, a, b)
   }
   |> Seq.maxBy(fun (x, _, _) -> x)

let _, a, b = answer

printfn "a = %d; b = %d" a b

//=================> Problem028 <=================================

let counter upto = 
   (Seq.unfold(fun (x, i, c) -> Some((x, i, c),(x+i, (if c%4=0 then i+2 else i), c+1))) (3, 2, 2)
   |> Seq.map(fun (x, _, _) -> x)
   |> Seq.take (upto * 2 - 2)
   |> Seq.sum) + 1

let answer = counter 1001

printfn "answer = %d" answer

//=================> Problem029 <=================================

let answer = seq {
                     for a in [2I..100I] do
                        for b in [2..100] do
                           yield a**b
                 }
                 |>Seq.distinct
                 |>Seq.length

printfn "answer = %d" answer

//=================> Problem030 <=================================

let fifthSum (n:float) = 
   Seq.map(fun x -> float(int(x)-0x30)**5.0) (string n)
   |> Seq.sum

let answer = 
   [2.0 .. 9.0**5.0*6.0]
   |> Seq.map(fun x -> (x, fifthSum x))
   |> Seq.filter(fun (x, y) -> x=y)
   |> Seq.sumBy fst

printfn "answer = %f" answer

//=================> Problem032 <=================================

let isPandigital n = 
   let str = new string(n.ToString().ToCharArray() |> Array.sort)
   "123456789" = str

let answer =
    seq {
        for x in [2L..100L] do
            let s =  if x > 9L then 123L else 1234L
            for y in [s..(10000L/(x+1L))] do
                let str = System.String.Concat(x, y, x*y) |> int64
                if isPandigital str then yield x*y
    }
    |> Seq.distinct
    |> Seq.sum

printfn "%i" answer

//=================> Problem034 <=================================

let factorial n =
   match n with
   | 0 -> 1
   | _ -> [1 .. n] |> Seq.reduce(fun acc n -> acc*n)


let sumFactorDigits n = 
    n.ToString()
    |> Seq.map(fun x -> factorial(int x - 0x30))
    |> Seq.sum

let answer = 
    [3..100000]
    |> Seq.map(fun x -> (x, sumFactorDigits x))
    |> Seq.filter(fun (x, y) -> x=y)
    |> Seq.sumBy fst

printfn "answer = %d" answer

//=================> Problem035 <=================================

let sieve limit =
   let mults p = p :: [ (p*p) .. p .. limit ] |> Set.ofList
   let prune y = Set.minElement y, Set.minElement y |> mults |> Set.difference y
   let rec loop (p, pset) ans =
      match pset with
      | x when x = Set.empty -> Set.add p ans
      | _ -> loop (prune pset) (Set.add p ans)
   loop ([ 3L..2L..limit ] |> Set.ofList |> prune) Set.empty |> Set.toList

let rotate arr = 
   seq {
      let len = Array.length arr
      for x in [0..len-1] do
      yield arr|>Array.permute(fun i -> (i + x) % len)
   }

let isPrime n = [|2L.. (n|>float|>sqrt|>int64)|] |> Array.exists(fun y -> n <> y && n % y=0L) |> not

let answer = 
   seq {yield 2L; yield! sieve 1000000L}
   |> Seq.map(fun x -> (x, x.ToString().ToCharArray() |> rotate |> Seq.map(fun x -> int64(new string(x)))))
   |> Seq.filter(fun (x, a) -> a |> Seq.forall isPrime)
   |> Seq.toArray
   |> Array.length

printfn "answer = %d" answer

//=================> Problem036 <=================================

let alpha = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
let rec changeBase num b =
   match num with
   | 0 -> ""
   | _ -> changeBase (num/b) b + string alpha.[num%b]

let isPalindromic str =
   str = new string (str.ToCharArray() |> Array.rev)

let answer = 
   [1..2..999999] //Skip even numbers. They always start with 1 and end in 0 in binary.
   |> Seq.map(fun x -> (x, changeBase x 2))
   |> Seq.filter(fun (x,y) -> isPalindromic (string x) && isPalindromic y)
   |> Seq.sumBy fst

printfn "answer = %d" answer

//=================> Problem037 <=================================

open System.Collections.Generic 

let sieve n = 
    seq { 
        yield 2 
        let knownComposites = new HashSet<int>() 
        for i in 3 .. 2 .. n do 
            let found = knownComposites.Contains(i) 
            if not found then 
                yield i 
            do for j in i .. i .. n do 
                   knownComposites.Add(j) |> ignore
   }

let primes = sieve 1000000 |> Seq.toList
let primeHashset = new HashSet<int>(primes)

let truncates n =
   let str = n.ToString()
   let rec loopLeft (s:string) = 
      match s with 
      | "" -> true
      | n when not(primeHashset.Contains(int n)) -> false
      | _ -> loopLeft(s.[.. s.Length-2])
   if not(loopLeft str) then false
   else
      let rec loopRight (s:string) = 
         match s with 
         | "" -> true
         | n when not(primeHashset.Contains(int n)) -> false
         | _ -> loopRight(s.[1..])
      if not(loopRight str) then false
      else true

let answer = 
   primes
   |> Seq.skipWhile(fun x -> x < 10)
   |> Seq.filter truncates
   |> Seq.take 11
   |> Seq.sum

printfn "answer = %d" answer

//=================> Problem038 <=================================

let isPandigital n = 
   let str = new string(n.ToString().ToCharArray() |> Array.sort)
   "123456789" = str

let countFactors (n:int) = 
      let rec loop (s:string, f:int) =
         let current = s + string(f*n)
         match current.Length with
         | l when l >= 9 -> if isPandigital current then Some(int current) else None
         | _ -> loop(current, f+1)
      loop("", 1)

let answer = 
   [2..10000]
   |> List.map countFactors
   |> Seq.choose(fun x -> x)
   |> Seq.max

printfn "answer = %d" answer

//=================> Problem039 <=================================

[<StructuralComparison; StructuralEquality>]
type Triangle =
    {
        a : float
        b : float
        c : float
        p : float
    }

let maximum = 1000.0

let rightTriangles = 
    seq {
            for a in [1.0 .. maximum] do
                for b in [1.0 .. maximum] do
                    let c = (a*a + b*b) ** 0.5
                    let p = a + b + c
                    if p <= maximum then
                        if a * a + b * b = c*c then
                            yield {a = a; b = b; c = c; p = p}
                        }
printfn "%f"
    (
        rightTriangles
        |> Seq.distinct
        |> Seq.groupBy(fun x -> x.p)
        |> Seq.maxBy(fun (x, y) -> y |> Seq.length)
        |> fst
    )

//=================> Problem040 <=================================

let sb = new System.Text.StringBuilder()

let superString = 
   [1..100000] |> Seq.fold(fun acc x -> sb.Append(x)) sb
   |> string

let d n = int(superString.[n-1]) - 0x30

let answer = d(1) * d(10) * d(100) * d(1000) * d(10000) * d(100000)

printfn "answer = %d" answer

//=================> Problem041 <=================================

open System.Collections.Generic 

let isPandigital n = 
   let str = new string(n.ToString().ToCharArray() |> Array.sort)
   let matcher = [1..str.Length] |> Seq.map string |> Seq.reduce(fun acc x -> acc+x)
   str = matcher

let sieve n = 
    seq { 
        yield 2 
        let knownComposites = new HashSet<int>() 
        for i in 3 .. 2 .. n do 
            let found = knownComposites.Contains(i) 
            if not found then 
                yield i 
            do for j in i .. i .. n do 
                   knownComposites.Add(j) |> ignore
   }

let primes = sieve 10000000 |> Seq.cache

let answer = 
   primes
   |> Seq.filter isPandigital
   |> Seq.max

printfn "answer = %d" answer

//=================> Problem042 <=================================

open System.IO

let t n = 0.5*n*(n+1.0)

let wordValue (str:string) = 
   str.ToCharArray()
   |> Seq.map(fun x -> int(x) - 64)
   |> Seq.sum
   |> float

let isTriangleNumber n = 
   seq {for x in [1.0..n] do if (t x) = n then yield n}
   |> Seq.length > 0

let answer = 
   File.ReadLines(".\\words.txt")
   |> Seq.map wordValue
   |> Seq.filter isTriangleNumber
   |> Seq.length

printfn "answer = %d" answer

//=================> Problem043 <=================================

let pandigitalNumbers = 
  let rec permutations list taken = 
      seq { if Set.count taken = Array.length list then yield [] else
            for l in list do
              if not (Set.contains l taken) then 
                for perm in permutations list (Set.add l taken)  do
                  yield l::perm }
  permutations ("0123456789".ToCharArray()) Set.empty 
  |> Seq.map(fun f -> new string(f |> List.toArray) |> int64)


let answer = 
    pandigitalNumbers
    |> Seq.sort
    |> Seq.nth 1000000

printfn "%i" answer

//=================> Problem044 <=================================

let p n = n*(n*3-1)/2

let pentNums = [for x in [1..10000] -> p x]

let answer =
   seq {
   for x in pentNums do
      for y in pentNums do
         let sum = x + y
         let difference = x-y
         if List.exists(fun x -> x=sum) pentNums && List.exists(fun x -> x=difference) pentNums then
            yield abs(x-y)
   }
   |> Seq.head

printfn "answer = %d" answer

//=================> Problem045 <=================================

//Skip all triangle numbers. All Triangle numbers are also Hexagonal numbers.

let pentagonal n = n * (3L*n-1L)/2L
let hexagonal n = n * (2L*n-1L)

let pentagonals = set [for n in [166L..100000L] -> pentagonal n]
let hexagonals = set [for n in [143L..100000L] -> hexagonal n]

let answer = 
   pentagonals
   |> Set.intersect hexagonals
   |> Set.minElement

printfn "answer = %d" answer

//=================> Problem046 <=================================

open System.Collections.Generic 

type Solution =
    {
        prime:int;
        square:int;
    }
    override this.ToString() = 
        System.String.Format("{0} + 2 * {1}²", this.prime, this.square)

let isPrime n = [|2.. (n|>float|>sqrt|>int)|] |> Array.exists(fun y -> n <> y && n % y=0) |> not

let odds = 
    Seq.unfold(fun x -> Some(x, x+2)) 1
    |> Seq.filter(fun x -> x |> isPrime |> not)

let sieve n = 
    seq { 
        yield 2 
        let knownComposites = new HashSet<int>() 
        for i in 3 .. 2 .. n do 
            let found = knownComposites.Contains(i) 
            if not found then 
                yield i 
            do for j in i .. i .. n do 
                   knownComposites.Add(j) |> ignore
   }

let primes = sieve 10000 |> Seq.cache

let solve n = 
    let rec exp i p = 
        match i with
        | x when x = n-p -> None
        | x when n = p + 2 * (i*i) -> Some({prime = p; square = i;})
        | _ -> exp (i+1) p
    primes
    |> Seq.takeWhile(fun x -> x < n)
    |> Seq.tryPick(fun p -> exp 1 p)

let answer = 
    odds
    |> Seq.skipWhile(fun x -> x |> solve |> Option.isSome)
    |> Seq.head

printfn "answer = %i" answer

//=================> Problem048 <=================================

let hugeSum =
   [1..1000]
   |> Seq.sumBy(fun x -> bigint x**x)
   |> string

let answer = hugeSum.[(hugeSum.Length - 10)..]
printfn "answer = %s" answer

//=================> Problem049 <=================================

open System.Collections.Generic 

let isPrime n = [|2.. (n|>float|>sqrt|>int)|] |> Array.exists(fun y -> n <> y && n % y=0) |> not

let sieve n = 
    seq { 
        yield 2 
        let knownComposites = new HashSet<int>() 
        for i in 3 .. 2 .. n do 
            let found = knownComposites.Contains(i) 
            if not found then 
                yield i 
            do for j in i .. i .. n do 
                   knownComposites.Add(j) |> ignore
   }

let arePermutations x y z =
   let difference = y - x
   if x = y || y = z || x = z then false
   elif z - y <> difference then false
   else
      let sortint n = new string(n.ToString() |> Seq.sort |> Seq.toArray)
      let xSort = sortint x
      let ySort = sortint y
      let zSort = sortint z
      xSort = ySort && ySort = zSort

let primes = sieve 9999 |> Seq.filter(fun x -> x > 1000) |> Seq.cache

let solutions = seq {
   for x in primes do
      for y in primes |> Seq.filter(fun f -> f > x) do
            let difference = y - x
            let z = y + difference
            if isPrime z && arePermutations x y z then yield (x,y,z)
   }

let str (x,y,z) = string x + string y + string z

let answer = 
   solutions
   |> Seq.filter(fun (x, _, _) -> x <> 1487)
   |> Seq.head
   |> str

printfn "answer = %s" answer

//=================> Problem050 <=================================

open System 
open System.Collections.Generic 

let sieve n = 
    seq { 
        yield 2 
        let knownComposites = new HashSet<int>() 
        for i in 3 .. 2 .. n do 
            let found = knownComposites.Contains(i) 
            if not found then 
                yield i 
            do for j in i .. i .. n do 
                   knownComposites.Add(j) |> ignore 
    }

let primes = sieve 1000000 |> Seq.toArray

let primeCounter n = 
    seq {
        for i in 1..50 do 
            let count =
                primes
                |> Seq.windowed i
                |> Seq.filter(fun x -> x |> Array.sum = n)
                |> Seq.map(fun x -> x |> Array.length)
            if count |> Seq.isEmpty then yield 0
            else yield count |> Seq.max
    }
    |> Seq.max

let answer = 
    primes
    |> Array.maxBy(primeCounter)

printfn "%i" answer

//=================> Problem052 <=================================

let sortDigits n =
   let str = n.ToString()
   new string(str.ToCharArray() |> Array.sort)

let areAllSame (n1, n2, n3, n4, n5) =
   n1 = n2 && n1 = n3 && n1 = n4 && n1 = n5

let answer = 
   Seq.unfold(fun x -> Some(x, x+1)) 123456
   |> Seq.filter(fun x -> areAllSame (sortDigits(x*2), sortDigits(x*3), sortDigits(x*4), sortDigits(x*5), sortDigits(x*6)))
   |> Seq.head

printfn "answer = %d" answer

//=================> Problem053 <=================================

let fact n =
   match n with
   | n when n = 0I -> 1I
   | _ -> [1I .. n] |> Seq.reduce(fun acc n -> acc*n)

let c n r = 
   (fact n) / (fact r * fact(n - r))

let answer = 
   seq {
      for n in [1I..100I] do
         for r in [1I..n] do
            yield c n r
   }
   |> Seq.filter(fun x -> x > 1000000I)
   |> Seq.length

printfn "answer = %d" answer

//=================> Problem054 <=================================

type Card = 
   {
      Value : int;
      Suit : char;
   }

let parseCard (str:string) =
   {
      Value =
         match str.[0] with
         | 'T' -> 10
         | 'J' -> 11
         | 'Q' -> 12
         | 'K' -> 13
         | 'A' -> 14
         | v -> int(v.ToString())
      Suit = str.[1] 
   };

type Hand(list : List<Card>) = 
   let cards = list |> List.sortBy(fun x -> x.Value) |> List.rev
   let singleGroupOfSize size =
      let grouping =
         cards
         |> List.toSeq
         |> Seq.groupBy(fun x -> x.Value)
         |> Seq.filter(fun (_, v) -> v |> Seq.length = size)
      if grouping |> Seq.length = 1 then Some(grouping |> Seq.head |> fst) else None

   member this.Cards
      with get() : List<Card> = cards

   member this.HighCard with get() : Card = cards |> List.head

   member this.HasHighest (other : Hand) : bool =
      let head = this.Cards
                  |> List.zip other.Cards
                  |> List.toSeq
                  |> Seq.skipWhile(fun (t, o) -> t.Value = o.Value)
                  |> Seq.head
      (head |> snd) > (head |> fst)

   member this.OnePair : int option =
      singleGroupOfSize 2

   member this.TwoPair : int option =
      let grouping = 
         this.Cards
         |> List.toSeq
         |> Seq.groupBy(fun x -> x.Value)
         |> Seq.filter(fun (_, v) -> v |> Seq.length = 2)
      if grouping |> Seq.length = 2 then Some(grouping |> Seq.maxBy(fun (x, _) -> x) |> fst) else None

   member this.ThreeOfAKind : int option =
      singleGroupOfSize 3

   member this.Straight : int option =
      let rec check list next original = 
         match list with
         | head :: tail when head.Value = next -> check tail (head.Value-1) original
         | [] -> Some(original)
         | _ -> None
      check this.Cards this.HighCard.Value this.HighCard.Value
      
   member this.Flush : int option =
      if this.Cards |> List.forall(fun x -> x.Suit = this.Cards.Head.Suit) then Some this.HighCard.Value else None

   member this.FullHouse : int option =
      let grouping = this.Cards |> List.toSeq |> Seq.groupBy(fun x -> x.Value)
      let two = grouping |> Seq.filter(fun (_, v) -> v |> Seq.length = 2)
      let three = grouping |> Seq.filter(fun (_, v) -> v |> Seq.length = 3)
      if two |> Seq.length = 1 && three |> Seq.length = 1 then Some(three |> Seq.map fst |> Seq.max) else None

   member this.FourOfAKind : int option =
      singleGroupOfSize 4

   member this.StraightFlush : int option =
      let s = [this.Straight; this.Flush];
      if s |> List.forall Option.isSome then s |> List.maxBy Option.get else None

   member this.RoyalFlush : int option =
      if this.StraightFlush |> Option.exists(fun x -> x = 14) then Some 14 else None
         
let compareHands (hand1 : (int * Hand)) (hand2 : (int * Hand)) : int =
   let compraro (applier : Hand -> int option) : int option =
      let firstHand, secondHand = (snd hand1, snd hand1 |> applier), (snd hand2, snd hand2 |> applier)
      if firstHand |> snd |> Option.isSome && secondHand |> snd |> Option.isSome then
         let firstValue = firstHand |> snd |> Option.get
         let secondValue = secondHand |> snd |> Option.get
         if firstValue = secondValue then
            if (fst firstHand).HasHighest (fst secondHand) then Some(fst hand1) else Some(fst hand2)
         elif firstValue > secondValue then Some(fst hand1)
         else Some(fst hand2)
      elif firstHand |> snd |> Option.isSome && secondHand |> snd |> Option.isNone then Some(fst hand1)
      elif firstHand |> snd |> Option.isNone && secondHand |> snd |> Option.isSome then Some(fst hand2)
      else None
   seq {
      yield compraro (fun x -> x.RoyalFlush)
      yield compraro (fun x -> x.StraightFlush)
      yield compraro (fun x -> x.FourOfAKind)
      yield compraro (fun x -> x.FullHouse)
      yield compraro (fun x -> x.Flush)
      yield compraro (fun x -> x.Straight)
      yield compraro (fun x -> x.ThreeOfAKind)
      yield compraro (fun x -> x.TwoPair)
      yield compraro (fun x -> x.OnePair)
      yield compraro (fun x -> Some(x.HighCard.Value))
   } |> Seq.pick(fun x -> x)
   

let hands = 
   System.IO.File.ReadLines(".\poker.txt")
   |> Seq.map(fun x -> x.Split(' '))
   |> Seq.map(fun x -> (
                           new Hand(x |> Seq.take 5 |> Seq.map parseCard |> Seq.toList),
                           new Hand(x |> Seq.skip 5 |> Seq.take 5 |> Seq.map parseCard |> Seq.toList);
                       ))
   |> Seq.filter (fun x -> compareHands (1, fst x) (2, snd x) = 1)

printfn "Player 1 winnings: %i" (hands |> Seq.length)

//=================> Problem055 <=================================

let isPalindrome n = 
   let str = n.ToString()
   str = new string(str.ToCharArray() |> Array.rev)

let revInt (n:bigint) = 
   bigint.Parse(new string(n.ToString().ToCharArray() |> Array.rev))

let isLychrel (n:bigint) = 
   let rec loop n i = 
      if i >= 50 then true
      else
         let rev = revInt(n)
         let n2 = rev + n
         if isPalindrome n2 then
            false
         else
            loop n2 (i+1)
   loop n 1

let answer = 
   [1I..9999I]
   |> List.filter(isLychrel)
   |> List.length

printfn "%d" answer

//=================> Problem056 <=================================

let answer = 
   seq {
         for a in [1I..99I] do
            for b in [1..99] do
               yield a**b
       }

   |> Seq.map(fun x -> x.ToString("R") |> Seq.sumBy(fun y -> int y - 0x30))
   |> Seq.max

printfn "answer = %d" answer

//=================> Problem057 <=================================

let generate =
    Seq.unfold (fun (n, d) -> Some((n, d), (d*2I+n, d+n))) (3I, 2I)

let answer = 
    generate
    |> Seq.take 1000
    |> Seq.filter(fun (n, d) -> n.ToString("R").Length > d.ToString("R").Length)
    |> Seq.length

printfn "%A" answer

//=================> Problem058 <=================================

let isPrime n =
    if n = 1L then false
    else [|2L.. (n|>float|>sqrt|>int64)|]
         |> Array.exists(fun y -> n <> y && n % y=0L)
         |> not


let answer = 
    let rec wrap (n, c, s, w, t, p) =
        match (p/t) with
        | a when a <= 0.1f && n > 1L -> w
        | _ ->
                let prime = if isPrime(n+s) then p+1.0f else p                 
                match c with
                | 4L -> wrap(n+s, 1L, s+2L, w, t+1.0f, prime)
                | 1L -> wrap(n+s, c+1L, s, w+2L, t+1.0f, prime)
                | _  -> wrap(n+s, c+1L, s, w, t+1.0f, prime)
    wrap(1L, 1L, 2L, 1L, 1.0f, 0.0f)


printfn "answer = %i" answer

//=================> Problem059 <=================================

let cipherData = 
   System.IO.File.ReadAllText(".\\cipher1.txt").Split(',')
   |> Seq.map(byte) |> Seq.cache

let passwords n =
   let max = int(ceil(float n/3.0))
   seq {
   for x in ['a'..'z'] do
      for y in ['a'..'z'] do
         for z in ['a'..'z'] do
            yield [byte(x);byte(y);byte(z)] |> List.replicate max |> List.concat
   }

let answer =
   seq {
      for pwd in passwords(Seq.length cipherData) do
         let decryptedBytes =
            cipherData
            |> Seq.zip pwd
            |> Seq.map(fun (x, m) -> x ^^^ m)
            |> Seq.toArray
         let text = System.Text.Encoding.ASCII.GetString(decryptedBytes)
         let numberOfSpaces = text |> Seq.filter(fun x -> x = ' ') |> Seq.length
         yield (numberOfSpaces, decryptedBytes)
   }
   |> Seq.maxBy fst
   |> snd
   |> Array.sumBy(fun x -> int(x))


printfn "answer = %d" answer

//=================> Problem063 <=================================

let len n =
   n.ToString().Length

let pow n p = 
   let rec loop nl c =
      if c = p then nl
      else nl * (loop nl (c+1L))
   loop n 1L

let answer =
   seq {
      for x in [8I .. 8I] -> seq { for y in [2..50] do if len(x ** y) = y then yield y }
   }
   //|> Seq.take 25
   |> Seq.iter(fun x -> printfn "pow = %A" x)

//=================> Problem074 <=================================

let factorial n = 
   match n with
   | 0UL -> 1UL
   | _ -> [1UL .. n] |> Seq.reduce(fun acc n -> acc*n)

let factDigits n = 
   n.ToString()
   |> Seq.map(fun x -> int x - 0x30)
   |> Seq.sumBy(fun x -> factorial(uint64 x))

let countFactorial n = 
   let factChain =  Set.singleton n
   let rec loop x l = 
      if l |> Set.count > 60 then
         0
      else
         let factDigit = factDigits x
         if l |> Set.exists(fun y -> y=factDigit) then
            l |> Set.count
         else
            let newChain = l + Set.singleton factDigit
            loop factDigit newChain
   loop n factChain

let answer = 
   Async.Parallel [ for i in 1UL..999999UL -> async { return countFactorial i } ]
   |> Async.RunSynchronously
   |> Seq.filter(fun x -> x = 60)
   |> Seq.length

printfn "answer = %d" answer

//=================> Problem081 <=================================

open System.IO

let matrix = array2D (File.ReadAllLines("matrix.txt") 
                      |> Array.map (fun l -> l.Split(',') |> Array.map int32))

let size = matrix |> Array2D.length1

let sum = 
    Array2D.init size size (fun i j -> if i = 0 && j = 0 then matrix.[i, j] else 0)

let top i j = sum.[i - 1, j] + matrix.[i, j]
let left i j = sum.[i, j - 1] + matrix.[i, j]



for i = 0 to size - 1 do
    for j = 0 to size - 1 do
        match (i, j) with
        | (0, 0) -> ()
        | (_, 0) -> sum.[i, j] <- top i j
        | (0, _) -> sum.[i, j] <- left i j
        | (_, _) -> sum.[i, j] <- min (top i j) (left i j)

let answer = sum.[size - 1, size - 1]

printfn "%i" answer

//=================> Problem092 <=================================

open System
open System.Threading
open System.Threading.Tasks

let sumDigitSquares n =
      n.ToString().ToCharArray()
      |> Array.map(fun x -> (float x - 48.0)**2.0)
      |> Array.sum
      |> int

let findDigitSquare n = 
   let rec loop l = 
      match l with
      | 1 -> false
      | 89 -> true
      | x -> loop(sumDigitSquares(x))
   loop n

let actImpl n pls loc = 
   if findDigitSquare n then loc + 1
   else loc

let counter = ref 0

Parallel.For(1, 10000000, (fun () -> 0), actImpl, (fun n -> Interlocked.Add(counter, n) |> ignore)) |> ignore

printfn "answer = %d" !counter

//=================> Problem096 <=================================

open System.IO
open System.Text.RegularExpressions

let tm apply t =
    (t |> fst |> apply, t |> snd |> apply)

module Array =
    let last (a:'T[]) = a.[(a |> Array.length) - 1]

let problems =
    Regex.Split(File.ReadAllText("sudoku.txt"), @"^Grid \d\d", RegexOptions.Multiline)
    |> Array.filter(fun x -> x.Trim().Length > 0)
    |> Array.map(fun x -> 
                        let y = Regex.Split(x.Trim(), @"\r\n")
                                |> Array.map(fun x -> x.ToCharArray() |> Array.map(fun x -> int(x) - 0x30))
                        array2D(y))

let numbers = [|1..9|] |> Set.ofArray

let flatten (a:'T[,]) =
    seq { for y in a -> y :?> 'T }
    |> Seq.toArray

let quadrant (a:int[,]) (x:int, y:int) =
    let inline n nz = (nz / 3) * 3
    Array2D.init 3 3 (fun i j -> a.[n y + i, n x + j])

let axes (a:int[,]) (x:int, y:int) =
    let xa = Array2D.init 1 9 (fun i j -> a.[y, j]) |> flatten
    let ya = Array2D.init 9 1 (fun i j -> a.[i, x]) |> flatten
    (xa, ya)

let solutions (a:int[,]) (x:int, y:int) = 
    let quad = quadrant a (x,y)
    let quadExclude = flatten quad |> Array.filter(fun x -> x <> 0)
    let axesExclude = 
        let a = axes a (x, y)
        (fst a) |> Array.append (snd a)
    let excludes = quadExclude |> Array.append axesExclude |> Set.ofArray
    numbers - excludes |> Set.toArray

//Fill in all of the places that require no guessing. This can also be used
//On complex problems to reduce the complexity for spaces that have known answers
let simpleSolve (a:int[,]) =
    let a = a |> Array2D.copy
    let w = (a |> Array2D.length1)-1
    let h = (a |> Array2D.length2)-1
    let rec doSolve (a:int[,]) = 
        let mutable affected = 0
        for x in [0.. w] do
            for y in [0.. h] do
                if a.[y,x] = 0 then
                    let sol = solutions a (x, y)
                    if sol |> Array.length = 1 then
                        a.[y, x] <- sol.[0]
                        affected <- affected + 1
        if affected = 0 then a else doSolve a
    doSolve a

let getZeros (a:int[,]) =
    seq {
    let w = (a |> Array2D.length1)-1
    let h = (a |> Array2D.length2)-1
    for x in [0.. w] do
            for y in [0.. h] do
                if a.[y,x] = 0 then yield (x,y)
    }


//Verify it is solved correctly
//Has overhead. Review candidate.
let isSolved (a:int[,]) = 
    let w = (a |> Array2D.length1) - 1
    let h = (a |> Array2D.length2) - 1
    let mutable pass = true
    for x in [0..w] do
        for y in [0..h] do
            let coordinate = (x, y)
            let quad = quadrant a coordinate |> flatten |> Set.ofArray = numbers
            let ax = axes a coordinate |> tm (fun x -> x |> Set.ofArray = numbers)
            pass <- pass && quad && fst ax && snd ax
    pass

let incrementWorkState(ws:(int * array<int>)[]) = 
    let rec inc (i:int) = 
        let item = ws.[i]
        let current = fst item
        let possibles = snd item
        let last = possibles |> Array.last
        if current <> last then
            let next = possibles.[(possibles |> Array.findIndex(fun x -> x = current))+1]
            let newItem = (next, possibles)
            printfn "%A" newItem
            ws.[i] <- newItem
    inc 0
        

let complexSolve (a:int[,]) = 
    let a = a |> Array2D.copy
    let zeros = getZeros a
                |> Seq.map(fun x -> (x, solutions a x)) 
                |> Seq.toArray
                |> Array.sortBy(fun (_, x) -> x |> Array.length)
    let workState = zeros |> Array.map(fun ((x, y), a) -> (a.[0], a))
    incrementWorkState(workState)
    
let answers = 
    problems
    |> Seq.skip 1
    |> Seq.map complexSolve
    |> Seq.head

printfn "%A" (answers)

//=================> Problem097 <=================================

let fastPow(x:bigint,n:bigint) =
   let mutable result = 1I
   let mutable xm = x
   let mutable nm = n
   while not nm.IsZero do
      if not nm.IsEven then
         result <- result * xm
         nm <- nm-1I
      xm <- xm**2
      nm <- nm/2I
   result

let answer = 
   let str = (28433I*fastPow(2I,7830457I)+1I).ToString("R")
   str.[(str.Length - 10)..]

printfn "answer = %s" answer

//=================> Problem099 <=================================

open System.IO


let parseLine (str:string[]) = 
    match str with
    | [|a; b|] ->  (a |> float, b |> float)
    | _ -> failwith("Unexpected line")

let answer = 
    File.ReadLines("base_exp.txt")
    |> Seq.mapi(fun i x -> (i, x.Split(',') |> parseLine))
    |> Seq.map(fun (i, (n, e)) -> (i, e *  log n))
    |> Seq.maxBy snd
    |> fst

printfn "answer = %i" answer

//=================> Problem102 <=================================

type Point = {x: int; y:int; }

let parseTriangle (str:string) = 
   let points = str.Split(',') |> Array.map(fun x -> int(x))
   let a = { x = points.[0]; y = points.[1]}
   let b = { x = points.[2]; y = points.[3]}
   let c = { x = points.[4]; y = points.[5]}
   [|a;b;c;|]

let testTriangle (poly:array<Point>, pt:Point) = 
   let mutable p1 = {x = 0; y = 0};
   let mutable p2 = {x = 0; y = 0};
   let mutable inside = false
   let mutable oldPoint = {x = poly.[poly.Length - 1].x; y = poly.[poly.Length - 1].y}
   for i in [0..poly.Length-1] do
      let newPoint = { x = poly.[i].x; y = poly.[i].y }
      if newPoint.x > oldPoint.x then
         p1 <- oldPoint
         p2 <- newPoint
      else
         p2 <- oldPoint
         p1 <-  newPoint
      if (newPoint.x < pt.x) = (pt.x <= oldPoint.x) && (pt.y - p1.y) * (p2.x - p1.x) < (p2.y - p1.y) * (pt.x - p1.x) then
         inside <- not(inside)
      oldPoint <- newPoint
   inside

let answer = 
   System.IO.File.ReadLines(".\\triangles.txt")
   |> Seq.map(parseTriangle)
   |> Seq.filter(fun x -> testTriangle(x, {x=0; y=0}))
   |> Seq.length

printfn "answer = %d" answer

//=================> Problem112 <=================================

let isBouncy n = 
   let nstr = n.ToString() |> Seq.map(fun x -> int x - 0x30)
   let rec loop(s:List<_>, lower, upper) = 
      match s with
      | _ when lower&&upper -> true
      | (x,y) :: tail when x > y -> loop(tail, lower, true)
      | (x,y) :: tail when x < y -> loop(tail, true, upper)
      | _ :: tail -> loop(tail, lower, upper)
      | [] -> lower&&upper
   loop(nstr |> Seq.pairwise |> Seq.toList, false, false)

let answer =
   let rec loop s numBouncy = 
      match s with
      | _ when numBouncy/decimal (s)=0.99M -> s
      | s when isBouncy s -> loop (s+1) (numBouncy+1M)
      | _  -> loop (s+1) numBouncy
   loop 101 -1M

printfn "answer = %d" answer

//=================> problem145 <=================================

open System
open System.Threading
open System.Threading.Tasks

let reverseNumber n = 
   new string(n.ToString().ToCharArray() |> Array.rev)

let allDigitsOdd n = 
   if n%2=0 then false
   else
      n.ToString()
      |> Seq.map(fun x -> int x - 0x30)
      |> Seq.forall(fun x -> x%2<>0)

let actImpl n pls loc = 
      if n%10 = 0 then loc
      else
         let reverse = reverseNumber n
         let sum = n + int(reverseNumber n)
         if sum%2<>0 && reverse.[0] <> '0' && allDigitsOdd sum then
            loc + 1
         else
            loc

let counter = ref 0

Parallel.For(1, 1000000000, (fun () -> 0), actImpl, (fun n -> Interlocked.Add(counter, n) |> ignore)) |> ignore

printfn "answer = %d" !counter

//=================> Problem162 <=================================

let max = System.UInt64.MaxValue;

let answer =
    1UL
    |> Seq.unfold(fun x -> Some(x, x+1UL))
    |> Seq.map(fun x -> (x, x.ToString("X")))
    |> Seq.takeWhile(fun (_, x) -> x.Length <= 16)
    |> Seq.filter(fun (_, x) -> x.Contains("A") && x.Contains("0") && x.Contains("F"))
    |> Seq.sumBy(fst)

printfn "answer = %x" answer

//=================> Problem206 <=================================

open System.Text.RegularExpressions

let reg = new Regex(@"^1\d2\d3\d4\d5\d6\d7\d8\d9\d0$", RegexOptions.Compiled)

let answer = 
   seq {1000000000UL..10UL..1389026623UL}
   |> Seq.filter(fun x -> reg.IsMatch((x*x).ToString()))
   |> Seq.head

printfn "answer = %d" answer

//=================> Problem357 <=================================

open System.Collections.Concurrent
open System
open System.Threading
open System.Threading.Tasks

let divisors x =
   [1..x|>float|>sqrt|>int]
   |> Seq.filter(fun y -> x%y=0)
   |> Seq.map(fun y -> [y;x/y])
   |> Seq.concat
   |> Seq.filter(fun y -> x<>y)

let set = new ConcurrentDictionary<int, bool>()
let isPrime n =
    let value = ref false
    if set.TryGetValue(n, value) then
        !value
    else
        let prime = [|2.. (n|>float|>sqrt|>int)|] |> Array.exists(fun y -> n <> y && n % y=0) |> not
        set.TryAdd(n, prime) |> ignore
        prime

let answer = ref 0
let count = ref 0
let watch = System.Diagnostics.Stopwatch.StartNew()

let actImpl n pls loc = 
    if !count % 100000 = 0 && !count <> 0 then
        printfn "%f complete" (float(!count) / 100000000.0 * 100.0)
        printfn "%A elapsed" (watch.Elapsed)
    let result = divisors(n) |> Seq.exists(fun z -> isPrime (z+n/z) |> not) |> not
    Interlocked.Increment(count) |> ignore
    match result with
    | true -> n
    | _ -> 0

let merge n =
    Interlocked.Add(answer, n) |> ignore

let options = new ParallelOptions()
options.MaxDegreeOfParallelism <- 8
Parallel.For(1, 100000001, options, (fun () -> 0), actImpl, merge) |> ignore

printfn "%i" !answer


