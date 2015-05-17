using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Collections;

namespace FCSlib.Data {
  public static class Range {
    private static byte GetNext(byte val) { return (byte) (val + 1); }
    private static short GetNext(short val) { return (short) (val + 1); }
    private static int GetNext(int val) { return val + 1; }
    private static long GetNext(long val) { return val + 1; }
    private static sbyte GetNext(sbyte val) { return (sbyte) (val + 1); }
    private static ushort GetNext(ushort val) { return (ushort) (val + 1); }
    private static uint GetNext(uint val) { return val + 1; }
    private static ulong GetNext(ulong val) { return val + 1; }
    private static double GetNext(double val) { return val + 1; }
    private static float GetNext(float val) { return val + 1; }
    private static decimal GetNext(decimal val) { return val + 1; }
    private static DateTime GetNext(DateTime val) { return val.AddDays(1); }
    private static char GetNext(char val) { return (char) (val + 1); }



    public static Range<byte> Create(byte start, byte end) {
      return new Range<byte>(start, end, GetNext);
    }
    public static Range<short> Create(short start, short end) {
      return new Range<short>(start, end, GetNext);
    }
    public static Range<int> Create(int start, int end) {
      return new Range<int>(start, end, GetNext);
    }
    public static Range<long> Create(long start, long end) {
      return new Range<long>(start, end, GetNext);
    }
    public static Range<sbyte> Create(sbyte start, sbyte end) {
      return new Range<sbyte>(start, end, GetNext);
    }
    public static Range<ushort> Create(ushort start, ushort end) {
      return new Range<ushort>(start, end, GetNext);
    }
    public static Range<uint> Create(uint start, uint end) {
      return new Range<uint>(start, end, GetNext);
    }
    public static Range<ulong> Create(ulong start, ulong end) {
      return new Range<ulong>(start, end, GetNext);
    }
    public static Range<double> Create(double start, double end) {
      return new Range<double>(start, end, GetNext);
    }
    public static Range<float> Create(float start, float end) {
      return new Range<float>(start, end, GetNext);
    }
    public static Range<decimal> Create(decimal start, decimal end) {
      return new Range<decimal>(start, end, GetNext);
    }
    public static Range<char> Create(char start, char end) {
      return new Range<char>(start, end, GetNext);
    }
    public static Range<DateTime> Create(DateTime start, DateTime end) {
      return new Range<DateTime>(start, end, GetNext);
    }

    public static Range<T> Create<T>(T start, T end, Func<T, T> getNext, Comparison<T> compare) {
      return new Range<T>(start, end, getNext, compare);
    }
    public static Range<T> Create<T>(T start, T end, Func<T, T> getNext) {
      return new Range<T>(start, end, getNext);
    }

  }

  public class Range<T> : IRange<T> {
    public Range(T start, T end, Func<T, T> getNext, Comparison<T> compare) {
      this.start = start;
      this.end = end;
      this.compare = compare;
      this.sequence = Functional.Sequence<T>(getNext, start, v => compare(getNext(v), end) > 0);
    }

    public Range(T start, T end, Func<T, T> getNext) : this(start, end, getNext, Compare) { }

    private static int Compare<U>(U one, U other) {
      return Comparer<U>.Default.Compare(one, other);
    }

    readonly Comparison<T> compare;
    readonly IEnumerable<T> sequence;

    readonly T start;
    public T Start {
      get { return start; }
    }

    readonly T end;
    public T End {
      get { return end; }
    }

    public bool Contains(T value) {
      return compare(value, start) >= 0 && compare(end, value) >= 0;
    }

    IEnumerator<T> IEnumerable<T>.GetEnumerator( ) {
      return sequence.GetEnumerator( );
    }

    IEnumerator IEnumerable.GetEnumerator( ) {
      return ((IEnumerable<T>) this).GetEnumerator( );
    }
  }
}
