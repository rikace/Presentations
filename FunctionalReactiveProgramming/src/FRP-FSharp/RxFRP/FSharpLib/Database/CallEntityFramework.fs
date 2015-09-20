module CallEntityFramework

open Microsoft.FSharp.Linq
open Microsoft.FSharp.Linq.Query
open Microsoft.FSharp.Linq.QuotationEvaluation

type EntityFrameworkMoq() =
    member x.GetId(id:int) =
        id 

type CallEntityFramework() =
    let asyncCall(id:int) =
        let perform = async {
            let ctx = EntityFrameworkMoq()
            let emp = Query.query<@ ctx.GetId(id) @>
            return emp
        }
        let res = Async.RunSynchronously(perform)
        res