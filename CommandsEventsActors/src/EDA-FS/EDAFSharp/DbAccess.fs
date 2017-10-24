module DbAccess

open System
open FSharp.Data.TypeProviders

[<Literal>]
let connectionString ="Server=DESKTOP-0G0BUJ0;Initial Catalog=StoreContext;Integrated Security=true;MultipleActiveResultSets=true"

type internal EntityConnection = SqlEntityConnection<ConnectionString=connectionString>
type sqlStore = SqlDataConnection<ConnectionString=connectionString>
let sb = sqlStore.GetDataContext()
