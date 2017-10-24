using System;
using System.Reflection;
using sys = System.Runtime.Caching;


namespace DemoEDAFSharp.Infrastructure.Cache
{
    public class MemoryCache : CacheBase
    {
        private sys.MemoryCache _cache;
        public DateTime AbsoluteExpiry { get; set; }
        public TimeSpan SlidingExpiry { get; set; }

        protected override void SetInternal(string key, object value)
        {
            var policy = new sys.CacheItemPolicy();
            Set(key, value, policy);
        }

        protected override void SetInternal(string key, object value, TimeSpan lifespan)
        {
            var policy = new sys.CacheItemPolicy();
            policy.SlidingExpiration = lifespan;
            Set(key, value, policy);
        }

        protected override void SetInternal(string key, object value, DateTime expiresAt)
        {
            var policy = new sys.CacheItemPolicy();
            policy.AbsoluteExpiration = expiresAt;
            Set(key, value, policy);
        }

        private void Set(string key, object value, sys.CacheItemPolicy policy)
        {
            _cache.Set(key, value, policy);
        }

        protected override object GetInternal(string key)
        {
            return _cache[key];
        }

        protected override bool ExistsInternal(string key)
        {
            return _cache.Contains(key);
        }

        protected override void RemoveInternal(string key)
        {
            if (Exists(key))
            {
                _cache.Remove(key);
            }
        }

        public override void Initialise()
        {
            if (_cache == null)
            {
                _cache = new sys.MemoryCache(Assembly.GetExecutingAssembly().FullName);
            }
        }
    }
}