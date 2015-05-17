module DbAccess

open System
open Microsoft.FSharp.Data.TypeProviders
open FSharp.Data.Experimental

[<Literal>]
let connectionString ="Server=.\SqlExpress;Initial Catalog=StoreContext;Integrated Security=SSPI;MultipleActiveResultSets=true"
    
type internal EntityConnection = SqlEntityConnection<ConnectionString=connectionString>   
type sqlStore = SqlDataConnection<ConnectionString=connectionString>
let sb = sqlStore.GetDataContext()
