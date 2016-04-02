using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using FuzzyMatch;
using System.IO;
using CommonHelpers;
using System.Threading;
using System.Net.Http;

namespace ParallelizingFuzzyMatch
{
    class Program
    {
        static void Main(string[] args)
        {

            var loadWords = Data.Words;

            var p = new ProcessFuzzyMacth();

            p.SequentialFuzzyMatch();
            p.ThreadFuzzyMatch();
            p.TwoThreadsFuzzyMatch();
            p.MultipleThreadsFuzzyMatch();
            p.ParallelLoopFuzzyMatch();
            p.MultipleTasksFuzzyMatch();
          //  p.LinqFuzzyMatch();
            p.ParallelLinqFuzzyMatch();
            

            var color = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n\nPress <Enter> to Continue.");
            Console.ForegroundColor = color;
            Console.ReadLine();

            var a = new AsyncLoadFiles();
            var matches = a.LoadDataMultipleParallelAndProcessAsync().Result;
            foreach (var match in matches)
            {
                Console.WriteLine(match);
            }

            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("\n\nPress <Enter> to exit.");
            Console.ReadLine();
        }
    }
}
