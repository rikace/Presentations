
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FCSlib.Data {
  public interface IRange<T> : IEnumerable<T> {
    T Start { get; }
    T End { get; }
    bool Contains(T value);
  }
}
