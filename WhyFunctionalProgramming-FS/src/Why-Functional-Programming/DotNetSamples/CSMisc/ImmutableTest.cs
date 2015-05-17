using System;
using System.Linq;
using System.Collections.Generic;

namespace CSMisc
{

	public class ImmutableTest
	{
		public List<int> MakeList() 
		{
			return new List<int> {1,2,3,4,5,6,7,8,9,10};
		}

		// Now let me test it:
		public void Test() 
		{ 
			var immutableSample = new ImmutabillitySample ();
			var odds = immutableSample.OddNumbers(MakeList()); 
			var evens = immutableSample.EvenNumbers(MakeList());

			// assert odds = 1,3,5,7,9 
			// assert evens = 2,4,6,8,10 
			Console.WriteLine ("Odd test passed: {0}", odds.Count == 5); 
			Console.WriteLine ("Even test passed: {0}", evens.Count == 5); 
		}

		// Everything works great, and the test passes, 
		// but I notice that I am creating the list twice
		// I should refactor this out, and here’s the new improved version:
		public void RefactoredTest() 
		{ 
			var list = MakeList();

			var immutableSample = new ImmutabillitySample ();
			var odds = immutableSample.OddNumbers(list); 
			var evens = immutableSample.EvenNumbers(list);

			// assert odds = 1,3,5,7,9 
			// assert evens = 2,4,6,8,10 
			Console.WriteLine ("Odd Refactored test passed: {0}", odds.Count == 5); 
			Console.WriteLine ("Even Refactored test passed: {0}", evens.Count == 5); 
		}

		// Why would a refactoring break the test? 
		// The list is mutable, and it is probable that the OddNumbers 
		// function is making destructive changes to the list as part of its filtering logic. 

		// When I call the OddNumbers function, 
		// I am unintentionally creating undesirable 

		//		SIDE EFFETCS !!
	}


}

