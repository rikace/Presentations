using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace FCSlib.Data {
  public interface IMemory<P, R> {
    bool HasResultFor(P val);
    R ResultFor(P val);
    void Remember(P val, R result);
  }
}
