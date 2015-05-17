using System;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Collections.Generic;
using System.Collections;
using System.Diagnostics;

namespace FCSlib.Data.Collections {
  public sealed class RedBlackTree<T> : IEnumerable<T> {
    public enum Color {
      Red,
      Black
    }

    private readonly bool isEmpty;
    public bool IsEmpty { get { return isEmpty; } }
    private readonly Color nodeColor;
    public Color NodeColor { get { return nodeColor; } }

    private readonly RedBlackTree<T> left;
    public RedBlackTree<T> Left {
      get {
        return left;
      }
    }
    private readonly RedBlackTree<T> right;
    public RedBlackTree<T> Right {
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

    public static readonly RedBlackTree<T> Empty = new RedBlackTree<T>( );

    #region Constructors
    private RedBlackTree( )  {
      isEmpty = true;
    }
 
    public RedBlackTree(Color nodeColor, RedBlackTree<T> left, T value, RedBlackTree<T> right) {
      this.nodeColor = nodeColor;
      this.left = left;
      this.right = right;
      this.value = value;
    }
        
    #endregion

    #region Balance
    private static RedBlackTree<T> Balance(Color nodeColor,
      RedBlackTree<T> left, T value, RedBlackTree<T> right) {
      if (nodeColor == RedBlackTree<T>.Color.Black) {
        if (!(left.IsEmpty) &&
          left.NodeColor == RedBlackTree<T>.Color.Red &&
          !(left.Left.IsEmpty) &&
          left.Left.NodeColor == RedBlackTree<T>.Color.Red)
          return new RedBlackTree<T>(Color.Red,
            new RedBlackTree<T>(Color.Black,
              left.Left.Left, left.Left.Value, left.Left.Right),
            left.Value,
            new RedBlackTree<T>(Color.Black,
              left.Right, value, right));
        if (!(left.IsEmpty) &&
          left.NodeColor == RedBlackTree<T>.Color.Red &&
          !(left.Right.IsEmpty) &&
          left.Right.NodeColor == RedBlackTree<T>.Color.Red)
          return new RedBlackTree<T>(Color.Red,
            new RedBlackTree<T>(Color.Black,
              left.Left, left.Value, left.Right.Left),
            left.Right.Value,
            new RedBlackTree<T>(Color.Black,
              left.Right.Right, value, right));
        if (!(right.IsEmpty) &&
          right.NodeColor == RedBlackTree<T>.Color.Red &&
          !(right.Left.IsEmpty) &&
          right.Left.NodeColor == RedBlackTree<T>.Color.Red)
          return new RedBlackTree<T>(Color.Red,
            new RedBlackTree<T>(Color.Black,
              left, value, right.Left.Left),
            right.Left.Value,
            new RedBlackTree<T>(Color.Black,
              right.Left.Right, right.Value, right.Right));
        if (!(right.IsEmpty) &&
          right.NodeColor == RedBlackTree<T>.Color.Red &&
          !(right.Right.IsEmpty) &&
          right.Right.NodeColor == RedBlackTree<T>.Color.Red)
          return new RedBlackTree<T>(Color.Red,
            new RedBlackTree<T>(Color.Black,
              left, value, right.Left),
            right.Value,
            new RedBlackTree<T>(Color.Black,
              right.Right.Left, right.Right.Value, right.Right.Right));
      }

      return new RedBlackTree<T>(nodeColor, left, value, right);
    }
    #endregion

    #region Contains
    public static bool Contains(T value, RedBlackTree<T> tree) {
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
      return RedBlackTree<T>.Contains(value, this);
    }
    #endregion

    #region Inserting

    public static RedBlackTree<T> Insert(T value, RedBlackTree<T> tree) {
      Func<RedBlackTree<T>, RedBlackTree<T>> ins = null;
      ins = t => {
        if (t.IsEmpty)
          return new RedBlackTree<T>(Color.Red, Empty, value, Empty);
        var compareResult = Comparer<T>.Default.Compare(value, t.Value);
        if (compareResult < 0)
          return Balance(t.NodeColor, ins(t.Left), t.Value, t.Right);
        else if (compareResult > 0)
          return Balance(t.NodeColor, t.Left, t.Value, ins(t.Right));
        else
          return t;
      };

      var insResult = ins(tree);
      return new RedBlackTree<T>(Color.Black, insResult.Left, insResult.Value, insResult.Right);
    }

    public RedBlackTree<T> Insert(T value) {
      return RedBlackTree<T>.Insert(value, this);
    }


    #endregion

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
      string colStr = NodeColor == Color.Black ? "B" : "R";
      if (IsEmpty)
        return String.Format("[{0} Empty]", colStr);
      return String.Format("[{0} {1} {2} {3}]", colStr, Left, Value, Right);
    }
  }
}
