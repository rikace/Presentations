using System.ComponentModel.Composition;
using System.ComponentModel.Composition.Hosting;
using System.Reflection;
using System.Web.Mvc;
using System.Web.Routing;
using Domain.DataAccess;

namespace DemoEDA
{
    public class MEFControllerFactory : DefaultControllerFactory
    {
        private static readonly CompositionContainer container;


        static MEFControllerFactory()
        {
            var catalog = new AggregateCatalog();
            catalog.Catalogs.Add(new AssemblyCatalog(Assembly.GetExecutingAssembly()));
            catalog.Catalogs.Add(new AssemblyCatalog(typeof (StoretUnitOfWork).Assembly));

            container = new CompositionContainer(catalog);
        }

        public static CompositionContainer Container
        {
            get { return container; }
        }

        public override IController CreateController(RequestContext requestContext,
            string controllerName)
        {
            IController controller = base.CreateController(requestContext, controllerName);

            container.ComposeParts(controller);

            return controller;
        }
    }
}