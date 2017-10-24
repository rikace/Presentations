using System;
using Domain.Entities;

namespace DemoEDAFSharp.Commands
{
    public class AddProductCommand : Command
    {
        private readonly string _cartId;
        private readonly int _quantity;

        public AddProductCommand(string cartId, Product product, int quantity)
        {
            _cartId = cartId;
            _quantity = quantity;
            Product = product;
            Id = Guid.NewGuid();
        }

        public Product Product { get; private set; }

        public int Quantity
        {
            get { return _quantity; }
        }

        public string CartId
        {
            get { return _cartId; }
        }
    }

    public class RemoveProductCommand : Command
    {
        private readonly string _cartId;

        public RemoveProductCommand(string cartId, Product product)
        {
            _cartId = cartId;
            Product = product;
            Id = Guid.NewGuid();
        }

        public Product Product { get; private set; }

        public string CartId
        {
            get { return _cartId; }
        }
    }

    public class SubmitOrderCommand : Command
    {
        private readonly string _cartId;

        public SubmitOrderCommand(string cartId, Order order)
        {
            _cartId = cartId;
            Order = order;
            Id = Guid.NewGuid();
        }

        public Order Order { get; private set; }

        public string CartId
        {
            get { return _cartId; }
        }
    }

    public class EmptyCardCommand : Command
    {
        private readonly string _cartId;

        public EmptyCardCommand(string cartId)
        {
            Id = Guid.NewGuid();
            _cartId = cartId;
        }

        public string CartId
        {
            get { return _cartId; }
        }
    }
}