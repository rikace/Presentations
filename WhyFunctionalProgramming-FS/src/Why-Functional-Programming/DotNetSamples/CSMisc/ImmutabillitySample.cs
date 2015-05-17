using System;
using System.Collections.Generic;

namespace CSMisc
{
	public class ImmutabillitySample
	{
		public List<int> OddNumbers(List<int> list) 
		{ 
			var lstOdd = new List<int> ();
			foreach(var n in list)
			{
				if(n % 2 != 0)
					lstOdd.Add(n);
			}
			list.Add (12);
			list.Add (14);
			return lstOdd;
		}

		public List<int> EvenNumbers(List<int> list) 
		{ 
			var lstEv = new List<int> ();
			foreach(var n in list)
			{
				if(n % 2 == 0)
					lstEv.Add(n);
			}


			return lstEv;
		}
	}
}

