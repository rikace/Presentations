namespace Easj360FSharp

    open System

    module Alghos =
        /// Computes factorial for the given integer.
        let rec fact n =
            match n with
            | 1 -> 1
            | n -> n * fact (n - 1)

        /// Computes fibonacci number for given integer.
        let rec fib n =
            match n with 
            | 1 -> 1
            | 2 -> 1
            | n -> fib (n - 1) + fib (n - 2)
    
        let measure_ms foo =   
            let tstart = System.DateTime.Now
            foo () |> ignore
            (System.DateTime.Now - tstart).TotalMilliseconds

          /// Measures average time taken by executing function passed as a parameter
          /// Function is executed 1000x to get better average result
        //let measure_ms_avg foo =   
        //     let tstart = DateTime.Now
        //     let rep = 1000
        //     for i = 1 to rep do
        //       foo 0 |> ignore
        //       done (System.DateTime.Now - tstart).TotalMilliseconds / float rep
    
        let measure msg foo =   
            let tstart = DateTime.Now
            let ret = foo ()
            let tdiff = DateTime.Now - tstart
            Console.WriteLine(msg, tdiff.TotalMilliseconds);
            ret

        let is_prime n =
            let max = int (Math.Sqrt( float n ))
            let anydiv = { 2 .. max } |> Seq.filter ( fun d -> n%d = 0) |> Seq.isEmpty
            not ((n = 1) || anydiv)

        let factorize primes n =
            let rec mul_count n d rep = 
              match (n%d) with 
                | 0 -> mul_count (n/d) d (rep + 1)
                | _ -> (rep, n)      
            let (fc, rem) = primes |> List.fold ( fun (fl, m) d ->
              let (rep, res) = mul_count n d 0
              match (rep) with | 0 -> (fl, m) | rep -> ((d, rep)::fl, res) ) ([], n)
            fc
    
        let factorize_slow n = factorize ([1 .. n] |> List.filter is_prime) n