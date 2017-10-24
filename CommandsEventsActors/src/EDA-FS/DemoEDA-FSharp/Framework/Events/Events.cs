using System;
using Domain.Entities;

namespace DemoEDAFSharp.Events
{
    public class OrderSubmittedEvent : Event
    {
        public OrderSubmittedEvent(Order order, Guid? id = null)
        {
            Order = order;
            Id = id ?? Guid.NewGuid();
        }

        public Order Order { get; private set; }
    }

    public class ProductAddedEvent : Event
    {
        public ProductAddedEvent(Product product, Guid? id = null)
        {
            Product = product;
            Id = id ?? Guid.NewGuid();
        }

        public Product Product { get; private set; }
    }

    public class ProductRemovedEvent : Event
    {
        public ProductRemovedEvent(Product product, Guid? id = null)
        {
            Product = product;
            Id = id ?? Guid.NewGuid();
        }

        public Product Product { get; private set; }
    }
}