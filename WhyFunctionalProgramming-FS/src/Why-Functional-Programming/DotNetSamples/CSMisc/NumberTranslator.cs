using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace CSMisc
{
	/* Some businesses advertise their phone number as a word, 
	 * phrase or combination of numbers and alpha characters. 
	 * This is easier for people to remember than a number. 
	 * You simply dial the numbers on the keypad that correspond 
	 * to the characters. For example, “1-800 FLOWERS” 
	 * translates to 1-800 3569377.
	 * 
	 * Write a simple program that translates a word 
	 * into a list of corresponding numbers */

	public class NumberTranslator
	{
		private readonly Dictionary<int, char[]> keys =
			new Dictionary<int, char[]>()
		{
			{1, new char[] {}},
			{2, new[] {'a', 'b', 'c'}},
			{3, new[] {'d', 'e', 'f'}},
			{4, new[] {'g', 'h', 'i'}},
			{5, new[] {'j', 'k', 'l'}},
			{6, new[] {'m', 'n', 'o'}},
			{7, new[] {'p', 'q', 'r', 's'}},
			{8, new[] {'t', 'u', 'v'}},
			{9, new[] {'w', 'x', 'y', 'z'}},
			{0, new[] {' '}},
		};

		// Translate method that takes a word and returns 
		// an array of corresponding numbers.
		// An imperative approach uses for-each loops 
		// and if-statements to iterate through characters
		// and populate an array of matching numbers

		public int[] Translate(string word)
		{
			var numbers = new List<int>();

			foreach (char character in word)
			{
				foreach(KeyValuePair<int, char[]> key in keys)
				{
					foreach(char c in key.Value)
					{
						if(c == character)
						{
							numbers.Add(key.Key);
						}
					}
				}
			}

			return (int[]) numbers.ToArray();
		}

		// Functional Approach 
		public int[] TranslateFunctional(string word)
		{
			return null;
		}
	}
}