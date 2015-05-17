using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FCSlib {
  public sealed class Lazy<T> {
    public Lazy(Func<T> function) {
      this.function = function;
    }

    public Lazy(T value) {
      this.value = value;
    }

    readonly Func<T> function;
    T value;
    bool forced;
    object forceLock = new object( );
    Exception exception;

    public T Force( ) {
      lock (forceLock) {
        return UnsynchronizedForce( );
      }
    }

    public T UnsynchronizedForce( ) {
      if (exception != null)
        throw exception;
      if (function != null && !forced) {
        try {
          value = function( );
          forced = true;
        }
        catch (Exception ex) {
          this.exception = ex;
          throw;
        }
      }
      return value;
    }

    public T Value {
      get { return Force( ); }
    }

    public bool IsForced {
      get { return forced; }
    }

    public bool IsException {
      get { return exception != null; }
    }

    public Exception Exception {
      get { return exception; }
    }
  }
}
