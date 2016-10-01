#r "../bin/PerfUtil.dll"

open System
open System.Threading

open PerfUtil

// basic usage

let result = Benchmark.Run <| repeat 100 (fun () -> Thread.Sleep 10)

// build a test setting

type IOperation =
    inherit ITestable
    abstract Run : unit -> unit

let dummy name (interval:int) = 
    {
        new IOperation with
            member __.Name = name
            member __.Init () = ()
            member __.Fini () = ()
            member __.Run () = System.Threading.Thread.Sleep(interval)
    }

let foo = dummy "foo" 10

// past version comparison

let tester = new PastImplementationComparer<IOperation>(foo, Version(0,1), historyFile = "D:/persist.xml", throwOnError = true)

tester.Run (repeat 100 (fun o -> o.Run()), id = "test 0")
tester.Run (repeat 100 (fun o -> o.Run()), id = "test 1")
tester.Run (repeat 100 (fun o -> o.Run()), id = "test 2")

tester.PersistCurrentResults()

// compare to other versions

let tester' = new ImplementationComparer<IOperation>(foo, [dummy "bar" 5 ; dummy "baz" 20 ], throwOnError = true)

tester'.Run (repeat 100 (fun o -> o.Run()), id = "test 0")