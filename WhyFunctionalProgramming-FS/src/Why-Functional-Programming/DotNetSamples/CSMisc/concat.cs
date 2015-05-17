using System;
using System.Text;

namespace CSMisc
{
	// the HConcat function is not a pure function, because 
	// it modifies the aMember data member in the class:
	// A pure function does not change any data outside of the function.
	public class Concat_1
	{
		private string aMember = "StringOne";

		public void Concat(string appendStr)
		{
			aMember += '-' + appendStr;
		}

		public void Test()
		{
			Concat("StringTwo");
			Console.WriteLine(aMember);
		}
	}


	// this same function is not pure because it modifies 
	// the contents of its parameter, sb.
	// This version of the program produces the same output 
	// as the first version, 
	// because the Concat function has changed the value (state) 
	// of its first parameter 
	// by invoking the Append member function
	public class Concat_2
	{
		public void Concat(StringBuilder sb, String appendStr)
		{
			sb.Append('-' + appendStr);
		}

		public void Test()
		{
			StringBuilder sb1 = new StringBuilder("StringOne");
			Concat(sb1, "StringTwo");
			Console.WriteLine(sb1);
		}
	}


	// This version of the program how to implement the Concat function 
	// as a pure function.
	// This version produces the same line of output: StringOne-StringTwo. 
	// Note that to retain the concatenated value, it is stored in the 
	// intermediate variable s2.
	public class Concat_3
	{
		public string Concat(string s, string appendStr)
		{
			return (s + '-' + appendStr);
		}

		public void Test()
		{
			string s1 = "StringOne";
			string s2 = Concat(s1, "StringTwo");
			Console.WriteLine(s2);
		}
	}
}