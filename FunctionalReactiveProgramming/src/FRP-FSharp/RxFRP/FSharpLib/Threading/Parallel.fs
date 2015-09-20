namespace Easj360FSharp

open System
open System.Threading

module Utils = begin            

  /// Same as List.fold_left function, but passes index of item in the list
  /// as a first parameter to fold function
  let list_fold_lefti (func:int -> 'a -> 'b -> 'a) (acc:'a) (l:'b list) =
    let rec fli (func:int -> 'a -> 'b -> 'a) (acc:'a) (l:'b list) (n:int) =
      match l with | [] -> acc | (h::t) -> fli func (func n acc h) t (n + 1)
    fli func acc l 0;
end

module ParallelList = begin

  /// Class that manages threads and executes operations in parallel
  type ('a, 'b) ThreadProcessor = class 
    val threads : Thread array;
    val worklist : 'a array;
    val results : 'b array;
    val result_states : int array;
    val operation : 'a -> 'b;
    val lock : obj;
    
    val mutable position : int;    
    val mutable running : bool;
    val mutable finished : int;
  
    /// Accepts number of threads, array of input data and 
    /// mapping function as parameters  
    new((count:int), (arr:'a array), (func:'a -> 'b)) as t = 
      { 
        lock = new obj();
        running = true; 
        worklist = arr; 
        results = Array.zeroCreate arr.Length;
        result_states = Array.create arr.Length 0;
        operation = func;
        position = 0;
        threads = Array.init count ( fun n -> new Thread(new ThreadStart(t.Process)) ); 
        finished = 0;
      }
      then 
        t.threads |> Array.iter ( fun t -> t.Start() );

    /// Stop execution and wait for all threads to complete
    member t.Stop () =
      t.running <- false;
      t.threads |> Array.iter ( fun t -> t.Join() );
      
    /// Private method executed by the thread(s)
    member t.Process () =
      // Get section in worklist that will be processed
      let (sf, st) = lock t.lock ( fun () -> 
        let len = t.worklist.Length;
        if (t.position = len) then
          t.running <- false; (1, 0)
        else
          let npos = t.position + (len - t.position) / (2*t.threads.Length);
          let ret = (t.position, npos);
          t.position <- npos + 1;
          ret )          
       // Process section
      for n = sf to st do 
        t.results.[n] <- (t.operation t.worklist.[n]);
        
        // TODO: Interlocked.Exchange(~&(t.result_states.[n]), 1) |> ignore;
        t.result_states.[n] <- 1;
        
      done;
      // Continue? tail-recursive...
      if (t.running) then t.Process();
      
    /// Reads results
    /// TODO: it could be better to return Seq<'b>
    member t.Results 
      with get() =
        let mutable res = [] in 
        let max = (t.worklist.Length - 1)
        let rec item n = 
          // TODO: Use Interlocked.Compare...
          if (t.result_states.[n] = 0) then Thread.Sleep(10); item n; else t.results.[n];
        for n = max downto 0 do
          res <- (item n)::res;
        done
        res;
  end

  let mutable threadcount = 2  
  let get_thread_count () = threadcount
  let set_thread_count n = threadcount <- n
  
  let filter (func:'a -> bool) (lst:'a list) = 
    let arr = lst |> List.toArray;
    let proc = new ThreadProcessor<'a, bool>(threadcount, arr, func)
    let ret = proc.Results |> Utils.list_fold_lefti ( fun i acc v -> 
      match v with | true -> (arr.[i])::acc | false -> acc ) []
    proc.Stop();
    ret
    
  let map (func:'a -> 'b) (lst:'a list) =
    let arr = lst |> List.toArray;
    let proc = new ThreadProcessor<'a, 'b>(threadcount, arr, func)
    let ret = proc.Results 
    proc.Stop();
    ret
       
end

//ParallelList.threadcount <- Environment.ProcessorCount / 2

/////////////////// TEST PARALLEL //////////////////////////
module testParallelList = 

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
      
    //// Console.WriteLine("PRIME_TEST [PARALLEL]");
    //let c = measure (" - time: {0}") ( fun () ->
    //    [1 .. 1000000]  |> ParallelList.filter is_prime |> List.length |> ignore 
    ////    Console.WriteLine(" - result: {0}", c) 
    //
    //let rec test_factorize () =
    //    let list_interval = [100000 .. 100015] 
    //    let print_it ((n:int), flist) =
    //      Console.Write(" - {0} = 1", n);
    //      flist |> List.iter ( fun (m, rep) -> Console.Write(" * {0}^{1}", m, rep); )
    //      Console.WriteLine();     
    //    
    //    Console.WriteLine("FACTORIZATION [PARALLEL]");
    //    measure (" - time: {0}") ( fun () ->
    // [1 .. 1000000] |> ParallelList.map ( fun n -> (n, factorize_slow n) ) |> List.iter print_it  
    //    
    ////    Console.WriteLine("COMBINATION [PARALLEL]");
    //let primes2 = measure " - searching: {0}" ( fun () -> [1 .. 1000000] |> ParallelList.filter is_prime ) |>
    //    measure " - factorizing: {0}" ( fun () -> [(1000000 - 500) .. 1000000] |> ParallelList.map (factorize primes) ) |> ignore 
    //    )