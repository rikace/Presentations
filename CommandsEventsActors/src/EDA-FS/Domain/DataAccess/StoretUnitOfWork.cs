using System;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using Domain.Common;
using Domain.Entities;

namespace Domain.DataAccess
{
    public interface IStoretUnitOfWork : IUnitOfWork
    {
        IRepository<Product> Products { get; }
        IRepository<Category> Categories { get; }
        IRepository<Cart> Carts { get; }
        IRepository<Order> Order { get; }
        IRepository<OrderDetail> OrderDetails { get; }
        void Commit();
        void Commit<TEvent>(Guid id, IEventStore<TEvent> eventStore) where TEvent : class;
    }

    [Export(typeof (IStoretUnitOfWork))]
    [PartCreationPolicy(CreationPolicy.NonShared)]
    public class StoretUnitOfWork : IStoretUnitOfWork
    {
        private ContextRepository<Cart> _cart;
        private ContextRepository<Category> _categories;
        private ContextRepository<OrderDetail> _orderDetails;
        private ContextRepository<Order> _orders;
        private ContextRepository<Product> _products;
        private StoreContext ctx;

        public StoretUnitOfWork()
        {
            ctx = new StoreContext();
            ctx.Configuration.LazyLoadingEnabled = true;
        }

        public void Dispose()
        {
            ctx.Dispose();
            ctx = null;
        }

        public IRepository<Product> Products
        {
            get { return _products ?? (_products = new ContextRepository<Product>(ctx)); }
        }

        public IRepository<Category> Categories
        {
            get { return _categories ?? (_categories = new ContextRepository<Category>(ctx)); }
        }

        public IRepository<Cart> Carts
        {
            get { return _cart ?? (_cart = new ContextRepository<Cart>(ctx)); }
        }

        public IRepository<Order> Order
        {
            get { return _orders ?? (_orders = new ContextRepository<Order>(ctx)); }
        }

        public IRepository<OrderDetail> OrderDetails
        {
            get { return _orderDetails ?? (_orderDetails = new ContextRepository<OrderDetail>(ctx)); }
        }

        public void Commit<TEvent>(Guid id, IEventStore<TEvent> eventStore) where TEvent : class
        {
            List<TEvent> events = eventStore.GetEvents(id);
            eventStore.SaveEvents(id, events);

            ctx.SaveChanges();
        }


        public void Commit()
        {
            ctx.SaveChanges();
        }

        public void Rollback()
        {
            // Not Implemented
        }
    }
}