using System;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Collections.Generic;
using System.Collections;

namespace FCSlib.Data.Collections {
  public sealed class UnbalancedBinaryTree<T> : IEnumerable<T> {
    private readonly bool isEmpty;
    public bool IsEmpty { get { return isEmpty; } }

    private readonly UnbalancedBinaryTree<T> left;
    public UnbalancedBinaryTree<T> Left {
      get {
        return left;
      }
    }
    private readonly UnbalancedBinaryTree<T> right;
    public UnbalancedBinaryTree<T> Right {
      get {
        return right;
      }
    }
    private readonly T value;
    public T Value {
      get {
        return value;
      }
    }

    public static readonly UnbalancedBinaryTree<T> Empty = new UnbalancedBinaryTree<T>( );


    #region Constructors
    public UnbalancedBinaryTree( ) {
      isEmpty = true;
    }
    
    public UnbalancedBinaryTree(UnbalancedBinaryTree<T> left, T value, UnbalancedBinaryTree<T> right) {
      this.left = left;
      this.right = right;
      this.value = value;
    }
    
    #endregion

    public static bool Contains(T value, UnbalancedBinaryTree<T> tree) {
      if (tree.IsEmpty)
        return false;
      else {
        int compareResult = Comparer<T>.Default.Compare(value, tree.Value);
        if (compareResult < 0)
          return Contains(value, tree.Left);
        else if (compareResult > 0)
          return Contains(value, tree.Right);
        else
          return true;
      }
    }

    public bool Contains(T value) {
      return UnbalancedBinaryTree<T>.Contains(value, this);
    }

    public static UnbalancedBinaryTree<T> Insert(T value, UnbalancedBinaryTree<T> tree) {
      if (tree.IsEmpty) {
        return new UnbalancedBinaryTree<T>(Empty, value, Empty);
      }
      else {
        int compareResult = Comparer<T>.Default.Compare(value, tree.Value);
        if (compareResult < 0)
          return new UnbalancedBinaryTree<T>(
            Insert(value, tree.Left),
            tree.Value,
            tree.Right);
        else if (compareResult > 0)
          return new UnbalancedBinaryTree<T>(
            tree.Left,
            tree.Value,
            Insert(value, tree.Right));
        else
          return tree;
      }
    }

    public UnbalancedBinaryTree<T> Insert(T value) {
      return UnbalancedBinaryTree<T>.Insert(value, this);
    }

    IEnumerator<T> System.Collections.Generic.IEnumerable<T>.GetEnumerator( ) {
      if (IsEmpty)
        yield break;

      foreach (T val in Left)
        yield return val;
      yield return Value;
      foreach (T val in Right)
        yield return val;
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator( ) {
      return ((IEnumerable<T>) this).GetEnumerator( );
    }

    public override string ToString( ) {
      return String.Format("[{0} {1} {2}]", Left, IsEmpty ? "Empty" : Value.ToString(), Right);
    }
  }
}
