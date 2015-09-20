namespace Easj360FSharp 

// #r "System.Core.dll"
// #r "FSharp.PowerPack.dll"
// #r "FSharp.PowerPack.Linq.dll"

open Microsoft.FSharp.Linq.QuotationEvaluation

module Quotation = 
    // Adds extension methods to Expr<_>


    // Evaluate a simple expression
    let x = <@ 1 + 2 * 3 @>
    
    let res = x.Eval()

    let ser f = 
        let serializer = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter()
        use mem = new System.IO.MemoryStream ()
        serializer.Serialize(mem, f)
        mem.Flush()
        mem.ToArray()
      
      // Quotations.Expr.Deserialize
    

    // Compile a function value expression
    let toUpperQuotation = <@ (fun (x : string) -> x.ToUpper()) @>
    let toUpperFunc = toUpperQuotation.Compile() ()

    toUpperFunc "don't panic"


