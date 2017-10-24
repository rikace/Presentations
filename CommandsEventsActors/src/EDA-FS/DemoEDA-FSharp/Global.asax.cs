using System;
using System.ComponentModel.Composition.Hosting;
using System.Reflection;
using System.Web;
using System.Web.Http;
using System.Web.Mvc;
using System.Web.Optimization;
using System.Web.Routing;
using Common.Framework;
using DemoEDA.App_Start;
using DemoEDAFSharp;
using DemoEDAFSharp.Commands;
using DemoEDAFSharp.Events;
using DemoEDAFsharp.Logging;
using Domain.DataAccess;
using DemoEDAFSharp.Infrastructure;

namespace DemoEDA
{
    public class MvcApplication : HttpApplication
    {
        protected void Application_Start()
        {
            SetMEF();

          //  DatabaseConfig.InitializeDatabases();
            AreaRegistration.RegisterAllAreas();
            WebApiConfig.Register(GlobalConfiguration.Configuration);
            FilterConfig.RegisterGlobalFilters(GlobalFilters.Filters);
            RouteConfig.RegisterRoutes(RouteTable.Routes);
            BundleConfig.RegisterBundles(BundleTable.Bundles);
            LogFactory.LogWith(new ConsoleLogger());
            RegisterEDA();
        }

        private void SetMEF()
        {
            var catalog = new AggregateCatalog();
            catalog.Catalogs.Add(new AssemblyCatalog(Assembly.GetExecutingAssembly()));
            catalog.Catalogs.Add(new AssemblyCatalog(typeof (IStoretUnitOfWork).Assembly));
            CompositionContainer container = MEFLoader.Init(catalog.Catalogs);
            DependencyResolver.SetResolver(new MefDependencyResolver(container));
        }

        private void RegisterEDA()
        {

            ISubscriber subscriber = ServiceLocator.Subscriber;
            // Commands
            var productCommandHandlers = new ProductCommandHandlers(ServiceLocator.EventBus, ServiceLocator.EventStore);
            subscriber.RegisterHandler(WrapLogger<AddProductCommand>(productCommandHandlers.Execute));
            subscriber.RegisterHandler(WrapLogger<RemoveProductCommand>(productCommandHandlers.Execute));
            subscriber.RegisterHandler(WrapLogger<SubmitOrderCommand>(productCommandHandlers.Execute));
            subscriber.RegisterHandler(WrapLogger<EmptyCardCommand>(productCommandHandlers.Execute));

            // Events
            var productEventHandlers = new ProductEventHandlers();
            subscriber.RegisterHandler(WrapLogger<ProductAddedEvent>(productEventHandlers.Handle));
            subscriber.RegisterHandler(WrapLogger<ProductRemovedEvent>(productEventHandlers.Handle));

            var logProductAdd = new LogForProduct();
            subscriber.RegisterHandler(WrapLogger<ProductAddedEvent>(logProductAdd.Handle));
            subscriber.RegisterHandler(WrapLogger<ProductRemovedEvent>(logProductAdd.Handle));

            var notifyWareHouse = new NotifyWareHouse();
            subscriber.RegisterHandler(WrapLogger<OrderSubmittedEvent>(notifyWareHouse.Handle));
            var sendConfirmationOrderShipped = new SendConfirmationOrderShipped();
            subscriber.RegisterHandler(WrapLogger<OrderSubmittedEvent>(sendConfirmationOrderShipped.Handle));

            var smailOrderConfirmation = new EmailOrderConfirmation();
            subscriber.RegisterHandler(WrapLogger<OrderSubmittedEvent>(smailOrderConfirmation.Handle));

            var notifySignalR = new NotifySignalR();
            subscriber.RegisterHandler(WrapLogger<OrderSubmittedEvent>(notifySignalR.Handle));
            subscriber.RegisterHandler(WrapLogger<ProductAddedEvent>(notifySignalR.Handle));
            subscriber.RegisterHandler(WrapLogger<ProductRemovedEvent>(notifySignalR.Handle));
        }

        Action<TMessage> WrapLogger<TMessage>(Action<TMessage> action)
        {
            return new LoggingExecutor<TMessage>(action).Handle;
        }
    }
}