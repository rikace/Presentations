module Template

// the template pattern takes three functions and forms a skeleton function named TemplateF
type Template(connF, queryF, disconnF) =     
    member this.Execute(conStr, queryStr) = 
        this.TemplateF conStr queryStr
    member this.TemplateF = 
            let f conStr queryStr = 
                connF conStr
                queryF queryStr
                disconnF ()
            f

// connect to the database
let connect conStr = 
    printfn "connect to database: %s" conStr

// query the database with the SQL query string
let query queryStr = 
    printfn "query database %s" queryStr

// disconnect from the database
let disconnect ()  = 
    printfn "disconnect"

let template() = 
    let s = Template(connect, query, disconnect)    
    s.Execute("<connection string>", "select * from tableA")

template()
