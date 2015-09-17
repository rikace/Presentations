module Utilitues
    module ConversionUtil = 
        
        let intToString(x:int) = x.ToString()

        let cprintfn c fmt = 
        Printf.kprintf
            (fun s -> 
                let orig = System.Console.ForegroundColor 
                System.Console.ForegroundColor <- c;
                System.Console.WriteLine(s)
                System.Console.ForegroundColor <- orig)
            fmt