using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace FCSlib.Data {
  public static class Memoizer<P, R> {
    static object memoryListLock = new object( );
    static Dictionary<string, IMemory<P, R>> memories;
    static Dictionary<string, IMemory<P, R>> Memories {
      get {
        lock (memoryListLock) {
          if (memories == null)
            memories = new Dictionary<string, IMemory<P, R>>( );
          return memories;
        }
      }
    }
    public static T CreateMemory<T>(string key) where T : IMemory<P, R>, new( ) {
      lock (memoryListLock) {
        if (Memories.ContainsKey(key))
          throw new InvalidOperationException("The memory key '" + key + "' is already in use.");
        T memory = new T( );
        memories[key] = memory;
        return memory;
      }
    }
    public static IMemory<P, R> CreateMemory(string key) {
      return CreateMemory<Memory<P, R>>(key);
    }
    public static IMemory<P, R> GetMemory(string key) {
      if (!(Memories.ContainsKey(key)))
        return CreateMemory(key);
      return Memories[key];
    }
  }
}