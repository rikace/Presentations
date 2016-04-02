using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Text.RegularExpressions;
using System.Diagnostics;

namespace WordsCounter
{
    class Program
    {
        public int WordCounter(string dirPath)
        {
            var wordCounter = new Dictionary<string, int>();

            foreach (string file in Directory.EnumerateFiles(dirPath, "*.*"))
            {
                var lines = File.ReadAllLines(file);
                foreach (var line in lines)
                {
                    var words = Regex.Split(line, @"\W+");
                    foreach (var word in words)
                    {
                        if (!string.IsNullOrEmpty(word) && word.Length > 0)
                        {
                            if (wordCounter.ContainsKey(word.ToUpper()))
                                wordCounter[word.ToUpper()]++;
                            else
                                wordCounter.Add(word.ToUpper(), 1);
                        }
                    }
                }
            }
            return wordCounter.Keys.Count;
        }


        public int WordCounterLinq(string dirPath)
        {
            var wordCounter = (from file in Directory.EnumerateFiles(dirPath, "*.*")
                               from line in File.ReadAllLines(file)
                               from word in Regex.Split(line, @"\W+")
                               where !string.IsNullOrEmpty(word) && word.Length > 0
                               select word.ToUpper()).GroupBy(s => s).ToDictionary(k => k.Key, v => v.Count());
            return wordCounter.Count;
        }
    
        // Test without filesystem
        // Parallelizable
        public Dictionary<string,int> LinesWordCounter(string[] lines)
        {
            var lineWordCounter = (from line in lines
                                   from word in Regex.Split(line, @"\W+")
                                    where !string.IsNullOrEmpty(word) && word.Length > 0
                                    select word.ToUpper()).GroupBy(s => s).ToDictionary(k => k.Key, v => v.Count());
            return lineWordCounter;
        }

        public int WordCounterFunctional(string dirPath)
        {
            var wordCounter = 
                               (from file in Directory.EnumerateFiles(dirPath, "*.*").AsParallel()
                                let lines = File.ReadAllLines(file)
                               select LinesWordCounter(lines));
            return wordCounter.SelectMany(p => p.Keys).Distinct().Count();
        }

        public int WordCounterPLinq(string dirPath)
        {
            var wordCounter = (from file in Directory.EnumerateFiles(dirPath, "*.*").AsParallel()
                               from line in File.ReadAllLines(file)
                               from word in Regex.Split(line, @"\W+")
                               where !string.IsNullOrEmpty(word) && word.Length > 0
                               select word.ToUpper()).GroupBy(s => s).ToDictionary(k => k.Key, v => v.Count());
            return wordCounter.Count;
        }



        static void Main(string[] args)
        { 
            var dataPath = "../../Shakespeare";

            var p = new Program();


            var sw = Stopwatch.StartNew();
            var wordsCount = p.WordCounter(dataPath);
            Console.WriteLine("WordCounter =>  {0} - {1} ms", wordsCount.ToString(), sw.ElapsedMilliseconds.ToString());
            sw.Restart();
            wordsCount = p.WordCounterLinq(dataPath);
            Console.WriteLine("WordCounter Linq =>  {0} - {1} ms", wordsCount.ToString(), sw.ElapsedMilliseconds.ToString());
            sw.Restart();
            wordsCount = p.WordCounterPLinq(dataPath);
            Console.WriteLine("WordCounter PLinq =>  {0} - {1} ms", wordsCount.ToString(), sw.ElapsedMilliseconds.ToString());
            sw.Restart();
            wordsCount = p.WordCounterFunctional(dataPath);
            Console.WriteLine("WordCounter FP =>  {0} - {1} ms", wordsCount.ToString(), sw.ElapsedMilliseconds.ToString());


            Console.WriteLine(wordsCount);
            

            Console.ReadLine();
        }
    }
}
