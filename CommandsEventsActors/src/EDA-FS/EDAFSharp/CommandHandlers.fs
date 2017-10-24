namespace EDAFSharp

open System
open Domain.Entities
open System.Threading
open System.Threading.Tasks
open System.Data.Linq
open System.Data.Entity
open Microsoft.FSharp.Data.TypeProviders
open FSharp.Data.SqlClient
open Domain.DataAccess
open System.Data.EntityClient


module CommandHandlers =

        let addProductCommand(id, product:Product, quantity) = async {
                                    let context = DbAccess.sqlStore.GetDataContext()
                                    let cartItem = query {  for cart in context.Carts do
                                                            where (cart.CartId = id && cart.ProductId = product.Id)
                                                            select cart } |> Seq.toList
                                    match cartItem with
                                    | []   -> let newCartItem = DbAccess.sqlStore.ServiceTypes.Carts()
                                              newCartItem.ProductId <- product.Id
                                              newCartItem.CartId <- string(id)
                                              newCartItem.Count <- quantity
                                              newCartItem.DateCreated <- DateTime.Now
                                              context.Carts.InsertOnSubmit(newCartItem)
                                              context.DataContext.SubmitChanges()
                                    | [cart] -> let cartQuery = query { for cart in context.Carts do
                                                                        where (cart.CartId = id && cart.ProductId = product.Id)
                                                                        select cart } |> Seq.head
                                                cartQuery.Count <- cartQuery.Count + quantity
                                                context.DataContext.SubmitChanges()
                                    }

        let removeProductCommand(id, product:Product) = async {
                                    let context = DbAccess.sqlStore.GetDataContext()
                                    let cartItem = query {  for cart in context .Carts do
                                                            where (cart.CartId = id && cart.ProductId = product.Id)
                                                            select cart } |> Seq.head
                                    match cartItem with
                                    | null ->   ()
                                    | cart when cart.Count > 1 -> cartItem.Count <- cartItem.Count - 1
                                    | cart -> context.Carts.DeleteOnSubmit(cart)
                                    context.DataContext.SubmitChanges() }

        let submitOrderCommand(id, order:Order) = async{
                                    let context = DbAccess.sqlStore.GetDataContext()
                                    let order' = DbAccess.sqlStore.ServiceTypes.Orders()
                                    order'.Address <- order.Address
                                    order'.City <- order.City
                                    order'.Country <- order.Country
                                    order'.State <- order.State
                                    order'.Email <- order.Email
                                    order'.FirstName <- order.FirstName
                                    order'.LastName <- order.LastName
                                    order'.Phone <- order.Phone
                                    order'.PostalCode <- order.PostalCode
                                    order'.Total <- order.Total
                                    order'.OrderDate <- DateTime.Now;
                                    context.Orders.InsertOnSubmit(order')
                                    context.DataContext.SubmitChanges()

                                    let carts = query {  for cart in context .Carts do
                                                            where (cart.CartId = id)
                                                            select cart } |> Seq.toList

                                    match carts with
                                    | [] -> ()
                                    | _  -> carts |> List.iter(fun c ->
                                                            let orderDetails = DbAccess.sqlStore.ServiceTypes.OrderDetails()
                                                            orderDetails.ProductId <- c.ProductId
                                                            orderDetails.OrderId <- order'.Id
                                                            orderDetails.Quantity <- c.Count
                                                            context.OrderDetails.InsertOnSubmit(orderDetails))
                                            context.Carts.DeleteAllOnSubmit(carts)
                                            context.DataContext.SubmitChanges()
                                    }

        let emptyCardCommand(id) =  async {
                                    let context = DbAccess.sqlStore.GetDataContext()
                                    let carts = query {  for cart in context .Carts do
                                                            where (cart.CartId = id)
                                                            select cart }
                                    context.Carts.DeleteAllOnSubmit(carts)
                                    context.DataContext.SubmitChanges() }