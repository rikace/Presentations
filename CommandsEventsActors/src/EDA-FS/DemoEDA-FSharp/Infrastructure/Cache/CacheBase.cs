using System;

namespace DemoEDAFSharp.Infrastructure.Cache
{
    public abstract class CacheBase : ICache
    {
        private CacheBase Current
        {
            get { return this; }
        }

        public abstract void Initialise();

        public void Set(string key, object value)
        {
            Current.SetInternal(key, value);
        }

        public T GetOrSet<T>(string key, Func<T> createValue, TimeSpan validFor) where T : class
        {
            if (Exists(key))
                return Current.GetInternal(key) as T;

            T value = createValue();
            Current.SetInternal(key, value, validFor);
            return value;
        }

        public void Set(string key, object value, DateTime expiresAt)
        {
            Current.SetInternal(key, value, expiresAt);
        }

        public void Set(string key, object value, TimeSpan validFor)
        {
            Current.SetInternal(key, value, validFor);
        }

        public object Get(string key)
        {
            return Current.GetInternal(key);
        }

        public void Remove(string key)
        {
            Current.RemoveInternal(key);
        }

        public bool Exists(string key)
        {
            return Current.ExistsInternal(key);
        }

        protected abstract void SetInternal(string key, object value);
        protected abstract void SetInternal(string key, object value, DateTime expiresAt);
        protected abstract void SetInternal(string key, object value, TimeSpan validFor);
        protected abstract object GetInternal(string key);
        protected abstract void RemoveInternal(string key);
        protected abstract bool ExistsInternal(string key);
    }
}