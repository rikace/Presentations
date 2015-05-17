
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace FCSlib.Data {
  public class Memory<P, R> : IMemory<P, R> {
    Dictionary<P, R> storage = new Dictionary<P, R>( );

    public bool HasResultFor(P val) {
      return storage.ContainsKey(val);
    }

    public R ResultFor(P val) {
      return storage[val];
    }

    public void Remember(P val, R result) {
      storage[val] = result;
    }
  }
}
