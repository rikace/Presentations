namespace PerfUtil.CaseStudy

    open System.Threading
    open System.IO

    open PerfUtil

    module Tests =

        type Marker = class end

        let inline roundtrip t (s : Serializer) = s.TestRoundTrip t

        let intlist = [1 .. 1000]

        [<PerfTest(1000)>]
        let ``int list`` (s : Serializer) = roundtrip intlist s


        let values = [1 .. 1000] |> List.map string

        [<PerfTest(1000)>]
        let ``string list`` (s : Serializer) = roundtrip values s


        let array3D = Array3D.init 100 100 100 (fun i j k -> float (i * j + k))

        [<PerfTest(10)>]
        let ``float [,,]`` s = roundtrip array3D s

        let objArray = [| obj() ; box "string" ; box (Some 53) ; box 2 |]

        [<PerfTest(1000)>]
        let ``obj []`` s = roundtrip objArray s


        let arrayDU = [| for i in 1 .. 10000 -> (Some ("lorem ipsum" + string i, i)) |]

        [<PerfTest(100)>]
        let ``(string * int) option []`` s =  roundtrip arrayDU s

        
        type BinTree<'T> = Leaf | Node of 'T * BinTree<'T> * BinTree<'T>

        let rec mkTree d =
            if d = 0 then Leaf
            else
                let b1 = mkTree (d-1)
                let b2 = mkTree (d-1)
                Node(d, b1, b2)


        let tree = mkTree 10

        [<PerfTest(100)>]
        let ``Binary Tree`` s = roundtrip tree s


        let largeQuotation =
            <@
                let rec fibAsync n =
                    async {
                        match n with
                        | _ when n < 0 -> return invalidArg "n" "negative inputs not supported."
                        | _ when n <= 1 -> return n
                        | n ->
                            let! fn = fibAsync (n-1)
                            let! fnn = fibAsync (n-2)
                            return fn + fnn
                    }

                fibAsync 42
            @>

        [<PerfTest(1000)>]
        let ``Large Quotation`` s = roundtrip largeQuotation s


        let set = set [ for i in 1 .. 1000 -> string i, i]

        [<PerfTest(100)>]
        let ``FSharp Set`` s = roundtrip set s