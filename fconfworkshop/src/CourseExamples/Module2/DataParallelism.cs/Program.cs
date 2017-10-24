using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;

namespace DataParallelism.CSharp
{
    static class Extensions
    {
        //public static ParallelQuery<TResult> MapReduce<TSource, TMapped, TKey, TResult>(
        //    this ParallelQuery<TSource> source,
        //    Func<TSource, IEnumerable<TMapped>> map,
        //    Func<TMapped, TKey> keySelector,
        //    Func<IGrouping<TKey, TMapped>, IEnumerable<TResult>> reduce)
        //{
        //    return source.SelectMany(map)
        //    .GroupBy(keySelector)
        //    .SelectMany(reduce);
        //}

        public static IEnumerable<string> EnumLines(this StringReader reader)
        {
            while (true)
            {
                string line = reader.ReadLine();
                if (null == line) yield break;

                yield return line;
            }
        }
    }

    public static class Program
    {
        static char[] delimiters = { ' ', ',', ';', ':', '\"', '.' };

        async static Task<string> ProcessBookAsync(
            string bookContent, string title, HashSet<string> stopwords)
        {
            using (var reader = new StringReader(bookContent))
            {
                var query = reader.EnumLines()
                    .AsParallel()
                    .SelectMany(line => line.Split(delimiters))
                    //.MapReduce(
                    //    map
                    //    group
                    //    reduce } }
                    //)
                    .ToList();

                //var words = query
                //    .Where(element =>
                //        !string.IsNullOrWhiteSpace(element.Word)
                //        && !stopwords.Contains(element.Word))
                //    .OrderByDescending(element => element.Count);

                var sb = new StringBuilder();

                sb.AppendLine($"'{title}' book stats");
                sb.AppendLine("Top ten words used in this book: ");
                //foreach (var w in words.Take(10))
                //    sb.AppendLine($"Word: '{w.Word}', times used: '{w.Count}'");

                sb.AppendLine($"Unique Words used: {query.Count()}");

                return sb.ToString();
            }
        }

        async static Task<string> DownloadBookAsync(string bookUrl)
        {
            using (var client = new HttpClient())
                return await client.GetStringAsync(bookUrl);
        }

        async static Task<HashSet<string>> DownloadStopWordsAsync()
        {
            string url =
                "https://raw.githubusercontent.com/6/stopwords/master/stopwords-all.json";

            using (var client = new HttpClient())
            {
                try
                {
                    var content = await client.GetStringAsync(url);
                    var words =
                        JsonConvert.DeserializeObject
                            <Dictionary<string, string[]>>(content);
                    return new HashSet<string>(words["en"]);
                }
                catch
                {
                    return new HashSet<string>();
                }

            }
        }

        static void RunBookMapReduce()
        {

            var booksList = new Dictionary<string, string>()
            {
                ["Moby Dick; Or, The Whale by Herman Melville"]
                = "http://www.gutenberg.org/cache/epub/2701/pg2701.txt",

                ["The Adventures of Tom Sawyer by Mark Twain"]
                = "http://www.gutenberg.org/cache/epub/74/pg74.txt",

                ["Treasure Island by Robert Louis Stevenson"]
                = "http://www.gutenberg.org/cache/epub/120/pg120.txt",

                ["The Picture of Dorian Gray by Oscar Wilde"]
                = "http://www.gutenberg.org/cache/epub/174/pg174.txt"
            };

            HashSet<string> stopwords = DownloadStopWordsAsync().GetAwaiter().GetResult();

            var output = new StringBuilder();

            Parallel.ForEach(booksList.Keys, key =>
            {
                var bookContent = DownloadBookAsync(booksList[key])
                    .GetAwaiter().GetResult();

                string result = ProcessBookAsync(bookContent, key, stopwords)
                    .GetAwaiter().GetResult();

                output.Append(result);
                output.AppendLine();
            });

            Console.Write(output.ToString());
            Console.ReadLine();
        }

        public static void Main(string[] args)
        {
            Console.WriteLine("WordCount");
            WordsCounterDemo.Demo();

            Demo.PrintSeparator();
            Console.WriteLine("parallel Reduce function implementation using Aggregate");

            int[] source = Enumerable.Range(0, 100000).ToArray();
            int result = source.AsParallel()
                    .Reduce((value1, value2) => value1 + value2);
        }
    }
}



