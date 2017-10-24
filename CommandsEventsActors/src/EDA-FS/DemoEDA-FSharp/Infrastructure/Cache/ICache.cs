using System;

namespace DemoEDAFSharp.Infrastructure.Cache
{
    public interface ICache
    {
        void Initialise();

        T GetOrSet<T>(string key, Func<T> createValue, TimeSpan validFor) where T : class;

        void Set(string key, object value);

        void Set(string key, object value, DateTime expiresAt);

        void Set(string key, object value, TimeSpan validFor);

        object Get(string key);

        void Remove(string key);

        bool Exists(string key);
    }
}