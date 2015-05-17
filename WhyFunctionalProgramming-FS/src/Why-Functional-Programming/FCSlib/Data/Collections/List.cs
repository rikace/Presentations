using System;
using System.Linq;
using System.Text;
using System.Reflection;

namespace FCSlib.Data.Collections {
  public sealed class List<T> : System.Collections.Generic.IEnumerable<T> {
    #region Head, Tail and IsEmpty
    private readonly T head;
    private readonly List<T> tail;
    private readonly bool isEmpty;
    public T Head {
      get {
        if (isEmpty)
          throw new InvalidOperationException("No head in an empty list");
        return head;
      }
    }
    public List<T> Tail {
      get {
        if (isEmpty)
          throw new InvalidOperationException("No tail in an empty list");
        return tail;
      }
    }
    public bool IsEmpty { get { return isEmpty; } }
    public static readonly List<T> Empty = new List<T>( );
    #endregion

    #region Cons
    public static List<T> Cons(T element, List<T> list) {
      if (list.IsEmpty)
        return new List<T>(element);
      else
        return new List<T>(element, list);
    }

    public List<T> Cons(T element) {
      return List<T>.Cons(element, this);
    }
    #endregion

    #region Appending
    // This recursive implementation is of course much more elegant,
    // but can result in stack overflows when the 'one' list is long
    public static List<T> AppendWithRecursion(List<T> one, List<T> other) {
      if (one.IsEmpty)
        return other;
      return Cons(one.Head, AppendWithRecursion(one.Tail, other));
    }

    public static List<T> Append(List<T> one, List<T> other) {
      if (one.IsEmpty)
        return other;
      List<T> newList = other;

      foreach (var element in one.Reverse( ))
        newList = newList.Cons(element);

      return newList;
    }

    public List<T> Append(List<T> other) {
      return List<T>.Append(this, other);
    }

    public List<T> AppendWithRecursion(List<T> other) {
      return List<T>.AppendWithRecursion(this, other);
    }
    #endregion

    #region Remove
    public static List<T> Remove(List<T> list, T element) {
      var memory = List<T>.Empty;
      var temp = list;
      while (!temp.IsEmpty && !System.Collections.Generic.EqualityComparer<T>.Default.Equals(temp.Head, element)) {
        memory = memory.Cons(temp.Head);
        temp = temp.Tail;
      }
      if (!temp.IsEmpty) {
        // forget the element itself
        temp = temp.Tail;
        // prepend the items we pushed aside
        foreach (var item in memory) {
          temp = temp.Cons(item);
        }
        return temp;
      }
      else
        // element wasn't found
        return list;
    }

    public List<T> Remove(T element) {
      return List<T>.Remove(this, element);
    }
    #endregion

    #region Constructors
    public List(T head, List<T> tail) {
      this.head = head;
      if (tail.IsEmpty)
        this.tail = List<T>.Empty;
      else
        this.tail = tail;
    }

    public List(T head) : this(head, List<T>.Empty) { }

    public List(T firstValue, params T[] values) {
      head = firstValue;
      if (values.Length > 0) {
        List<T> newtail = List<T>.Empty;
        for (int i = values.Length - 1; i >= 0; i--)
          newtail = newtail.Cons(values[i]);
        tail = newtail;
      }
      else
        tail = List<T>.Empty;
    }

    private List( ) {
      isEmpty = true;
    }

    public List(System.Collections.Generic.IEnumerable<T> source) {
      T[] sa = source.ToArray( );
      int sal = sa.Length;
      if (sal > 0) {
        head = sa[0];
        tail = List<T>.Empty;
        for (int i = sal - 1; i > 0; i--)
          tail = tail.Cons(sa[i]);
      }
      else
        isEmpty = true;
    }
    #endregion

    #region IEnumerable support
    public System.Collections.Generic.IEnumerator<T> GetEnumerator( ) {
      for (var element = this; element != List<T>.Empty; element = element.Tail)
        yield return element.Head;
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator( ) {
      return this.GetEnumerator( );
    }
    #endregion

    #region ToString
    public override string ToString( ) {
      var result = "[";
      if (!IsEmpty)
        result +=
        Functional.FoldL1(
          (r, x) => r + ", " + x,
          Functional.Map(x => x.ToString( ), this));
      result += "]";
      return result;
    }
    #endregion
  }
}