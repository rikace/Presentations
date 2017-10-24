using System.Collections.Generic;
using System.ComponentModel.Composition.Hosting;
using System.ComponentModel.Composition.Primitives;

namespace DemoEDA
{
    public static class MEFLoader
    {
        public static CompositionContainer Init(ICollection<ComposablePartCatalog> catalogParts = null)
        {
            var catalog = new AggregateCatalog();
            if (catalogParts != null)
                foreach (ComposablePartCatalog part in catalogParts)
                    catalog.Catalogs.Add(part);

            var container = new CompositionContainer(catalog);

            return container;
        }
    }
}