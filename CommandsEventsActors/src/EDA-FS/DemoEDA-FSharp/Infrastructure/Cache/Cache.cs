using System;
using System.Threading;

namespace DemoEDAFSharp.Infrastructure.Cache
{
    public class Cache
    {
        private static readonly Lazy<ICache> _instance = new Lazy<ICache>(() =>
        {
            var cache = new MemoryCache();
            cache.Initialise();
            return cache;
        }, LazyThreadSafetyMode.PublicationOnly);

        private Cache()
        {
        }

        public static ICache Instance
        {
            get { return _instance.Value; }
        }
    }
}