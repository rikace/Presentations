using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FRPFSharp;
using Microsoft.FSharp.Core;

namespace FRPCSharp
{
	[TestClass]
	public class Program
	{
		private static FSharpFunc<TA, FSharpFunc<TB, TC>> _<TA, TB, TC>(Func<TA, TB, TC> f)
		{
			return FSharpFunc<TA, FSharpFunc<TB, TC>>.FromConverter(
				a => FSharpFunc<TB, TC>.FromConverter(
					b => f(a, b)));
		}

		private static FSharpFunc<TA, TB> _<TA, TB>(Func<TA, TB> f)
		{
			return FSharpFunc<TA, TB>.FromConverter(a => f(a));
		}

		private static FSharpFunc<T, Unit> _<T>(Action<T> a)
		{
			return FSharpFunc<T, Unit>.FromConverter(t => { a(t); return null; });
		}

		[TestMethod]
		public void TestHold()
		{
			var e = EventContainer<int>.newDefault();
			Behavior<int> b = e.Event.Hold(0);
			var @out = new List<int>();
			Listener l = b.updates().Listen(Handler<int>.New(x => @out.Add(x)));
			e.send(2);
			e.send(9);
			l.Unlisten.Invoke(null);

			CollectionAssert.AreEqual(new[]
			{
				2,
				9
			}, @out.Select(x => (int)x).ToArray());
		}

 		[TestMethod]
		public void TestSnapshot()
		{
			var b = new BehaviorContainer<int>(0,0);
			var trigger = EventContainer<long>.newDefault();
			var @out = new List<string>();
			var l = trigger.Event.Snapshot(b.Behavior, _((long x, int y) => x + " " + y))
				.Listen(Handler<string>.New(x => @out.Add(x)));
			trigger.send(100L);
			b.send(2);
			trigger.send(200L);
			b.send(9);
			b.send(1);
			trigger.send(300L);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				"100 0",
				"200 2",
				"300 1"
			}, @out);
		}

		[TestMethod]
		public void TestApply()
		{
			var bf = new BehaviorContainer<FSharpFunc<long, String>>(0,_((long b) => "1 " + b));
			var ba = new BehaviorContainer<long>(0,5L);
			var @out = new List<String>();
			Listener l = Behavior<long>.apply<string>(bf.Behavior, ba.Behavior).Value().Listen(_((string x) => { @out.Add(x); }));
			bf.send(_((long b) => "12 " + b));
			ba.send(6L);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				"1 5",
				"12 5",
				"12 6"
			}, @out);
		}

		[TestMethod]
		public void TestLift()
		{
			var a = new BehaviorContainer<int>(0,1);
			var b = new BehaviorContainer<long>(0,5L);
			var @out = new List<String>();
			Listener l = Behavior<int>.lift0<long, string>(
				_((int x, long y) => x + " " + y),
				a.Behavior,
				b.Behavior
				).Value().Listen(_((string x) => { @out.Add(x); }));
			a.send(12);
			b.send(6L);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				"1 5",
				"12 5",
				"12 6"
			}, @out);
		}

		[TestMethod]
		public void TestValues()
		{
			var b = new BehaviorContainer<int>(0,9);
			var @out = new List<int>();
			var l = b.Behavior.Value().Listen(_((int x) => @out.Add(x)));
			b.send(2);
			b.send(7);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				9,
				2,
				7
			}, @out);
		}

		[TestMethod]
		public void TestValuesThenmerge()
		{
			var bi = new BehaviorContainer<int>(0,9);
			var bj = new BehaviorContainer<int>(0,2);
			var @out = new List<int>();
			Listener l = Event<int>.mergeWith(_((int x, int y) => x + y), bi.Behavior.Value(), bj.Behavior.Value())
				.Listen(_((int x) => { @out.Add(x); }));
			bi.send(1);
			bj.send(4);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				11,
				1,
				4
			}, @out);
		}

		[TestMethod]
		public void TestsendEvent()
		{
			var e = EventContainer<int>.newDefault();
			var @out = new List<int>();
			Listener l = e.Event.Listen(_((int x) => { @out.Add(x); }));
			e.send(5);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				5
			}, @out);
			e.send(6);
			CollectionAssert.AreEqual(new[]
			{
				5
			}, @out);
		}

		[TestMethod]
		public void TestmergeNonSimultaneous()
		{
			var e1 = EventContainer<int>.newDefault();
			var e2 = EventContainer<int>.newDefault();
			var @out = new List<int>();
			Listener l = Event<int>.merge(e1.Event, e2.Event).Listen(_((int x) => { @out.Add(x); }));
			e1.send(7);
			e2.send(9);
			e1.send(8);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				7,
				9,
				8
			}, @out);
		}

		[TestMethod]
		public void TestCoalesce()
		{
			var e1 = EventContainer<int>.newDefault();
			var e2 = EventContainer<int>.newDefault();
			var @out = new List<int>();
			Listener l =
				Event<int>.merge(e1.Event, Event<int>.merge(e1.Event.Map<int>(_((int x) => x * 100)), e2.Event))
					.Coalesce(_((int a, int b) => a + b))
					.Listen(_((int x) => { @out.Add(x); }));
			e1.send(2);
			e1.send(8);
			e2.send(40);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				202,
				808,
				40
			}, @out.ToArray());
		}


		[TestMethod]
		public void TestFilter()
		{
			var e = EventContainer<char>.newDefault();
			var @out = new List<char>();
			Listener l = e.Event.Filter(_((char c) => Char.IsUpper(c))).Listen(_((char c) => { @out.Add(c); }));
			e.send('H');
			e.send('o');
			e.send('I');
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				'H',
				'I'
			}, @out);
		}

		[TestMethod]
		public void TestAccum()
		{
			var ea = EventContainer<int>.newDefault();
			var @out = new List<int>();
			Behavior<int> sum = ea.Event.accum((int)100, _((int a, int s) => a + s));
			Listener l = sum.updates().Listen(_((int x) => { @out.Add(x); }));
			ea.send(5);
			ea.send(7);
			ea.send(1);
			ea.send(2);
			ea.send(3);
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				105,
				112,
				113,
				115,
				118
			}, @out);
		}

		class SB
		{
			public SB(string a, string b, Behavior<string> sw)
			{
				A = a;
				B = b;
				Sw = sw;
			}

			public readonly string A;
			public readonly string B;
			public readonly Behavior<string> Sw;
		}

		[TestMethod]
		public void TestSwitchB()
		{
			var esb = EventContainer<SB>.newDefault();
			// Split each field @out of SB so we can update multiple behaviours in a
			// single transaction.
			Behavior<string> ba = esb.Event.Map(_((SB s) => s.A)).FilterNotNull().Hold("A");
			Behavior<string> bb = esb.Event.Map(_((SB s) => s.B)).FilterNotNull().Hold("a");
			Behavior<Behavior<string>> bsw = esb.Event.Map(_((SB s) => s.Sw)).FilterNotNull().Hold(ba);
			Behavior<string> bo = Behavior<string>.SwitchB(bsw);
			var @out = new List<string>();
			Listener l = bo.Value().Listen(_((string c) => { @out.Add(c); }));
			esb.send(new SB("B", "b", null));
			esb.send(new SB("C", "c", bb));
			esb.send(new SB("D", "d", null));
			esb.send(new SB("E", "e", ba));
			esb.send(new SB("F", "f", null));
			esb.send(new SB(null, null, bb));
			esb.send(new SB(null, null, ba));
			esb.send(new SB("G", "g", bb));
			esb.send(new SB("H", "h", ba));
			esb.send(new SB("I", "i", ba));
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				"A",
				"B",
				"c",
				"d",
				"E",
				"F",
				"f",
				"F",
				"g",
				"H",
				"I"
			}, @out);
		}

		class SE
		{
			public SE(char a, char b, Event<char> sw)
			{
				A = a;
				B = b;
				Sw = sw;
			}

			public readonly char A;
			public readonly char B;
			public readonly Event<char> Sw;
		}

		[TestMethod]
		public void TestSwitchE()
		{
			var ese = EventContainer<SE>.newDefault();
			Event<char> ea = ese.Event.Map(_((SE s) => s.A)).FilterNotNull();
			Event<char> eb = ese.Event.Map(_((SE s) => s.B)).FilterNotNull();
			Behavior<Event<char>> bsw = ese.Event.Map(_((SE s) => s.Sw)).FilterNotNull().Hold(ea);
			var @out = new List<char>();
			Event<char> eo = Behavior<char>.SwitchE(bsw);
			Listener l = eo.Listen(_<char>(@out.Add));
			ese.send(new SE('A', 'a', null));
			ese.send(new SE('B', 'b', null));
			ese.send(new SE('C', 'c', eb));
			ese.send(new SE('D', 'd', null));
			ese.send(new SE('E', 'e', ea));
			ese.send(new SE('F', 'f', null));
			ese.send(new SE('G', 'g', eb));
			ese.send(new SE('H', 'h', ea));
			ese.send(new SE('I', 'i', ea));
			l.Unlisten.Invoke(null);
			CollectionAssert.AreEqual(new[]
			{
				'A',
				'B',
				'C',
				'd',
				'e',
				'F',
				'G',
				'h',
				'I'
			}, @out);
		}

		static void Main(string[] args)
		{

		}
	}
}
