using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using InteropLib;

namespace FPDemoCS
{
    class Program
    {
        static void Main(string[] args)
        {
            var five = new DollarsFS(5);
            // act
            DollarsFS result = five.Times(2);

            DollarsFS dollars2 = new DollarsFS(10);

            bool are10DollarsEquals = dollars2 == result;

            Console.WriteLine("Are 2 10 Dollars Equals = {0}", are10DollarsEquals);

            Console.ReadLine();
        }
    }
}
