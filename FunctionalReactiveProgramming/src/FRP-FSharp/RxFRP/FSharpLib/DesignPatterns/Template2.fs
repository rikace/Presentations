module Template2

// connection, query, and disconnect functions
let connect(conStr ) = printfn "connect using %s" conStr
let query(queryStr) = printfn "query with %s" queryStr
let disconnect() = printfn "disconnect"

// template pattern
let template(connect, query, disconnect) (conStr:string) (queryStr:string)= 
    connect(conStr)
    query(queryStr)
    disconnect()

// concrete query
let queryFunction = template(connect, query, disconnect)
// execute the query
do queryFunction "<connection string>" "select * from tableA"
