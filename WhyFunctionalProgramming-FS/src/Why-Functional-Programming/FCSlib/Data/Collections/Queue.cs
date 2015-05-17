using System;
using System.Linq;
using System.Text;
using System.Reflection;

namespace FCSlib.Data.Collections {
  public sealed class Queue<T> {
    private readonly List<T> f, r;

    public bool IsEmpty {
      get { return f.IsEmpty; }
    }

    public static readonly Queue<T> Empty = new Queue<T>( );
    private Queue(List<T> f, List<T> r) {
      this.f = f;
      this.r = r;
    }

    public Queue( )
      : this(List<T>.Empty, List<T>.Empty) {
    }

    public Queue(System.Collections.Generic.IEnumerable<T> source) {
      Queue<T> temp = Queue<T>.Empty;
      foreach (T item in source)
        temp = temp.Snoc(item);
      f = temp.f;
      r = temp.r;
    }

    public Queue(T first, params T[] values) {
      Queue<T> temp = Queue<T>.Empty;
      temp = temp.Snoc(first);
      foreach (T item in values)
        temp = temp.Snoc(item);
      f = temp.f;
      r = temp.r;
    }

    public static Queue<T> Snoc(Queue<T> q, T e) {
      return CheckBalance(new Queue<T>(q.f, q.r.Cons(e)));
    }

    public Queue<T> Snoc(T e) {
      return Snoc(this, e);
    }

    private static Queue<T> CheckBalance(Queue<T> q) {
      if (q.f.IsEmpty)
        return new Queue<T>(new List<T>(Functional.Reverse(q.r)), List<T>.Empty);
      else
        return q;
    }

    public T Head {
      get {
        return f.Head;
      }
    }

    public Queue<T> Tail {
      get {
        return CheckBalance(new Queue<T>(f.Tail, r));
      }
    }

    public override string ToString( ) {
      return String.Format("[f:{0} r:{1}]", f, r);
    }
  }
}