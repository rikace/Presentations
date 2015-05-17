using System;
using System.Linq;
using System.Text;
using System.Reflection;

namespace FCSlib.Data.Collections {
  public static class CollectionExtensions {
    public static List<T> ToList<T>(this System.Collections.Generic.IEnumerable<T> source) {
      return new List<T>(source);
    }
    public static Queue<T> ToQueue<T>(this System.Collections.Generic.IEnumerable<T> source) {
      return new Queue<T>(source);
    }
  }
}
