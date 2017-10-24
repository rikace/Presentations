using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using Domain.DataAccess;
using Domain.Entities;

namespace DemoEDAFSharp.Models
{
    public class ShoppingCart
    {
        public const string CartSessionKey = "CartId";

        private static IStoretUnitOfWork _storetUnitOfWork;

        private string ShoppingCartId { get; set; }

        public static ShoppingCart GetCart(HttpContextBase context, IStoretUnitOfWork storetUnitOfWork)
        {
            var cart = new ShoppingCart();
            _storetUnitOfWork = storetUnitOfWork;
            cart.ShoppingCartId = GetCartId(context);
            return cart;
        }

        public void AddToCart(Product product, int quantity)
        {
            Cart cartItem = _storetUnitOfWork.Carts.Find(
                c => c.CartId == ShoppingCartId && c.ProductId == product.Id).FirstOrDefault();

            if (cartItem == null)
            {
                cartItem = new Cart
                {
                    ProductId = product.Id,
                    CartId = ShoppingCartId,
                    Count = quantity,
                    DateCreated = DateTime.Now
                };

                _storetUnitOfWork.Carts.Add(cartItem);
            }
            else
            {
                cartItem.Count += quantity;
            }

            _storetUnitOfWork.Commit();
        }

        public void RemoveFromCart(int id)
        {
            Cart cartItem = _storetUnitOfWork.Carts.Find(
                cart => cart.CartId == ShoppingCartId
                        && cart.ProductId == id).FirstOrDefault();

            int itemCount = 0;

            if (cartItem != null)
            {
                if (cartItem.Count > 1)
                {
                    cartItem.Count--;
                    itemCount = cartItem.Count;
                }
                else
                {
                    _storetUnitOfWork.Carts.Remove(cartItem);
                }

                _storetUnitOfWork.Commit();
            }
        }

        public void EmptyCart()
        {
            IQueryable<Cart> cartItems = _storetUnitOfWork.Carts.Find(cart => cart.CartId == ShoppingCartId);

            foreach (Cart cartItem in cartItems)
            {
                _storetUnitOfWork.Carts.Remove(cartItem);
            }
            _storetUnitOfWork.Commit();
        }

        public List<Cart> GetCartItems()
        {
            return _storetUnitOfWork.Carts.Find(cart => cart.CartId == ShoppingCartId).ToList();
        }

        public int GetCount()
        {
            // Get the count of each item in the cart and sum them up
            int? count = _storetUnitOfWork.Carts.Find(c => c.CartId == ShoppingCartId).Sum(c => (int?) c.Count);
            return count ?? 0;
        }

        public decimal GetTotal()
        {
            // Multiply product price by count of that product to get 
            // the current price for each of those albums in the cart
            // sum all product price totals to get the cart total
            decimal? total =
                _storetUnitOfWork.Carts.Find(c => c.CartId == ShoppingCartId)
                    .Sum(c => (int?) c.Count*c.Product.Price);

            return total ?? decimal.Zero;
        }

        public void CreateOrder(Order order)
        {
            decimal orderTotal = 0;

            List<Cart> cartItems = GetCartItems();

            // Iterate over the items in the cart, adding the order details for each
            foreach (Cart item in cartItems)
            {
                var orderDetail = new OrderDetail
                {
                    ProductId = item.ProductId,
                    OrderId = order.Id,
                    UnitPrice = item.Product.Price,
                    Quantity = item.Count
                };

                // Set the order total of the shopping cart
                orderTotal += (item.Count*item.Product.Price);

                _storetUnitOfWork.OrderDetails.Add(orderDetail);
            }

            // Set the order's total to the orderTotal count
            order.Total = orderTotal;

            _storetUnitOfWork.Commit();
            EmptyCart();
        }

        // We're using HttpContextBase to allow access to cookies.
        public static string GetCartId(HttpContextBase context)
        {
            if (context.Session[CartSessionKey] == null)
            {
                Guid tempCartId = Guid.NewGuid();
                context.Session[CartSessionKey] = tempCartId.ToString();
            }
            return context.Session[CartSessionKey].ToString();
        }
    }
}