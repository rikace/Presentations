using System;
using System.Collections.Generic;
using System.Linq;
using System.Web.Mvc;
using Common.Framework;
using DemoEDA;
using DemoEDA.Infrastructure;
using DemoEDAFSharp.Events;
using Domain;
using Domain.DataAccess;
using Domain.Entities;

namespace DemoEDAFSharp.Commands
{
    public class ProductCommandHandlers :
        ICommandHandler<AddProductCommand>,
        ICommandHandler<RemoveProductCommand>,
        ICommandHandler<SubmitOrderCommand>,
        ICommandHandler<EmptyCardCommand>
    {
        private readonly IEventPublisher _eventPublisher;
        private readonly IEventStore<Event> _eventStore;
        private readonly IStoretUnitOfWork _storetUnitOfWork;

        public ProductCommandHandlers(IEventPublisher eventPublisher, IEventStore<Event> eventStore)
        {
            _eventStore = eventStore;
            _eventPublisher = eventPublisher;
            _storetUnitOfWork = (DependencyResolver.Current as MefDependencyResolver).GetService<IStoretUnitOfWork>();
        }

        public void Execute(AddProductCommand command)
        {
            Guard.NotNull(command, "command");
            Guard.NotNull(_storetUnitOfWork, "Repository is not initialized.");

            Product product = command.Product;
            string shoppingCartId = command.CartId;
            int quantity = command.Quantity;

            Cart cartItem = _storetUnitOfWork.Carts.Find(
                c => c.CartId == shoppingCartId && c.ProductId == product.Id).FirstOrDefault();

            if (cartItem == null)
            {
                cartItem = new Cart
                {
                    ProductId = product.Id,
                    CartId = shoppingCartId,
                    Count = quantity,
                    DateCreated = DateTime.Now
                };
                _storetUnitOfWork.Carts.Add(cartItem);
            }
            else
            {
                cartItem.Count += quantity;
            }

            _eventStore.SaveEvent(command.Id, new ProductAddedEvent(product));
            _storetUnitOfWork.Commit();
        }

        public void Execute(EmptyCardCommand command)
        {
            Guard.NotNull(command, "command");
            Guard.NotNull(_storetUnitOfWork, "Repository is not initialized.");

            IQueryable<Cart> cartItems = _storetUnitOfWork.Carts.Find(cart => cart.CartId == command.CartId);

            foreach (Cart cartItem in cartItems)
            {
                _storetUnitOfWork.Carts.Remove(cartItem);
            }
            _storetUnitOfWork.Commit();
        }

        public void Execute(RemoveProductCommand command)
        {
            Guard.NotNull(command, "command");
            Guard.NotNull(_storetUnitOfWork, "Repository is not initialized.");

            Product product = command.Product;
            string shoppingCartId = command.CartId;

            Cart cartItem = _storetUnitOfWork.Carts.Find(
                cart => cart.CartId == shoppingCartId
                        && cart.ProductId == product.Id).FirstOrDefault();

            if (cartItem != null)
            {
                if (cartItem.Count > 1)
                {
                    cartItem.Count--;
                }
                else
                {
                    _storetUnitOfWork.Carts.Remove(cartItem);
                }

                _storetUnitOfWork.Commit();
            }
            _eventStore.SaveEvent(command.Id, new ProductRemovedEvent(product));
        }

        public void Execute(SubmitOrderCommand command)
        {
            Guard.NotNull(command, "command");
            Guard.NotNull(_storetUnitOfWork, "Repository is not initialized.");

            Order order = command.Order;
            string shoppingCartId = command.CartId;

            order.OrderDate = DateTime.Now;
            _storetUnitOfWork.Order.Add(order);
            _storetUnitOfWork.Commit();

            decimal orderTotal = 0;
            List<Cart> cartItems = _storetUnitOfWork.Carts.Find(cart => cart.CartId == shoppingCartId).ToList();

            foreach (Cart item in cartItems)
            {
                var orderDetail = new OrderDetail
                {
                    ProductId = item.ProductId,
                    OrderId = order.Id,
                    UnitPrice = item.Product == null
                        ? _storetUnitOfWork.Products.FindById(item.ProductId).Price
                        : item.Product.Price,
                    Quantity = item.Count
                };
                orderTotal += (item.Count*item.Product.Price);
                _storetUnitOfWork.OrderDetails.Add(orderDetail);
            }

            order.Total = orderTotal;
            foreach (Cart cartItem in cartItems)
            {
                _storetUnitOfWork.Carts.Remove(cartItem);
            }

            _storetUnitOfWork.Commit();
            _eventPublisher.Publish(new OrderSubmittedEvent(order));
        }
    }
}