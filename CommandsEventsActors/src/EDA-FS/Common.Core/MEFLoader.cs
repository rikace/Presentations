using System;
using System.Collections.Generic;
using System.ComponentModel.Composition.Hosting;
using System.ComponentModel.Composition.Primitives;

namespace Common.Core
{
    public static class MEFLoader
    {
        public static CompositionContainer Init(ICollection<ComposablePartCatalog> catalogParts = null)
        {
            AggregateCatalog catalog = new AggregateCatalog();
            if (catalogParts != null)
                foreach (var part in catalogParts)
                    catalog.Catalogs.Add(part);

            CompositionContainer container = new CompositionContainer(catalog);

            return container;
        }
    }
}
