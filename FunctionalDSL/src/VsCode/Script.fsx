open System

// ------------------------------------------------------------------
// Domain model 

type [<Measure>] GBP

type Code = string
type Name = string
type UnitPrice = decimal<GBP>
type Quantity = decimal
type Amount = decimal<GBP>

type Product = Product of Code * Name * UnitPrice
type Category = Name * Product list

type TenderKind = Cash | Card

type SaleLineItem = 
  | SaleLineItem of Product * Quantity
  | TenderLineItem of TenderKind * Amount

type Operator = string
type Sale = Operator option * DateTime * SaleLineItem list
type History = Sale list

// ------------------------------------------------------------------
// Sample data

let products = 
  [ Product("1", "Tea", 1.45M<GBP>)
    Product("2", "Biscuits", 3.5M<GBP>)
    Product("3", "The Guardian", 4.0M<GBP>) ]

/// Lookup product in the database using a specified key
let findProduct key = 
  products |> Seq.tryFind (fun (Product(code, _, _)) ->
    code = key)

// Find product with the specified key and return sale line item
let key = "2"
let productOpt = findProduct key
match productOpt with 
| None -> ()
| Some product ->
    let sale = SaleLineItem(product, 1.0M)
    printfn "%A" sale

// TODO: Turn this sample into a console application


type Item = string * Status
and Status = Done | NotDone | Due of DateTime

type Todo = Item list



let getItemsOverDue (lst:Item list) (d:DateTime option)=
    let date = defaultArg d DateTime.Now
    lst
    |> List.filter(fun (str,status) -> 
        match status with
        | Status.Due dueDate when dueDate < date -> true
        | _ -> false)
    
