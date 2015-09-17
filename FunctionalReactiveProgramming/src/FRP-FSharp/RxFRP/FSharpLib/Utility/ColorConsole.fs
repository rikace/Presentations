module ColorConsole


let cprintfn c fmt = 
    Printf.kprintf
        (fun s -> 
            let orig = System.Console.ForegroundColor 
            System.Console.ForegroundColor <- c;
            System.Console.WriteLine(s)
            System.Console.ForegroundColor <- orig)
        fmt
