using System;
using System.Threading;
using Common.Framework;
using DemoEDAFSharp.Infrastructure;
using DemoEDAFsharp.Logging;
using Microsoft.AspNet.SignalR;

namespace DemoEDAFSharp.Events
{
    public class ProductEventHandlers :
        IEventHandler<ProductAddedEvent>,
        IEventHandler<ProductRemovedEvent>
    {
        public void Handle(ProductAddedEvent handle)
        {
            LogFactory.Logger.Info("Product {0} added", handle.Product.Name);
            Thread.Sleep(500);
        }

        public void Handle(ProductRemovedEvent handle)
        {
            LogFactory.Logger.Info("Product {0} removed", handle.Product.Name);
            Thread.Sleep(500);
        }
    }

    public class LogForProduct :
        IEventHandler<ProductAddedEvent>,
        IEventHandler<ProductRemovedEvent>
    {
        public void Handle(ProductAddedEvent handle)
        {
            LogFactory.Logger.Info("Product {0} added", handle.Product.Name);
        }

        public void Handle(ProductRemovedEvent handle)
        {
            LogFactory.Logger.Info("Product {0} removed", handle.Product.Name);
        }
    }

    public class NotifySignalR :
        IEventHandler<ProductAddedEvent>,
        IEventHandler<ProductRemovedEvent>,
        IEventHandler<OrderSubmittedEvent>
    {
        public void Handle(OrderSubmittedEvent handle)
        {
            IHubContext ctx = GlobalHost.ConnectionManager.GetHubContext<SignalRHub>();
            ctx.Clients.All.broadcastMessage(String.Format("Order Id {0} has been submited succesfully", handle.Order.Id));
        }

        public void Handle(ProductAddedEvent handle)
        {
            IHubContext ctx = GlobalHost.ConnectionManager.GetHubContext<SignalRHub>();
            ctx.Clients.All.broadcastMessage(String.Format("Product {0} has been added succesfully", handle.Product.Name));
        }

        public void Handle(ProductRemovedEvent handle)
        {
            IHubContext ctx = GlobalHost.ConnectionManager.GetHubContext<SignalRHub>();
            ctx.Clients.All.broadcastMessage(String.Format("Product {0} has been removed succesfully",
                handle.Product.Name));
        }
    }


    public class NotifyWareHouse :
        IEventHandler<OrderSubmittedEvent>
    {
        public void Handle(OrderSubmittedEvent handle)
        {
            // send email to WareHouse
            Console.WriteLine("Sending email to wareouse with order details");
        }
    }

    public class SendConfirmationOrderShipped :
        IEventHandler<OrderSubmittedEvent>
    {
        public void Handle(OrderSubmittedEvent handle)
        {
            Console.WriteLine("Sending confirmation order shipped");
        }
    }

    public class EmailOrderConfirmation :
        IEventHandler<OrderSubmittedEvent>
    {
        public void Handle(OrderSubmittedEvent handle)
        {
            Console.WriteLine("Sending confirmation order by email to customer");
        }
    }
}