//using Core.Common.Extensions;

using System;
using System.Collections.Generic;
using System.ComponentModel.Composition.Hosting;
using System.Web.Mvc;
using DemoEDA.Infrastructure;

namespace DemoEDA
{
    public class MefDependencyResolver : IDependencyResolver
    {
        private readonly CompositionContainer _Container;

        public MefDependencyResolver(CompositionContainer container)
        {
            _Container = container;
        }

        public object GetService(Type serviceType)
        {
            return _Container.GetExportedValueByType(serviceType);
        }

        public IEnumerable<object> GetServices(Type serviceType)
        {
            return _Container.GetExportedValuesByType(serviceType);
        }
    }
}