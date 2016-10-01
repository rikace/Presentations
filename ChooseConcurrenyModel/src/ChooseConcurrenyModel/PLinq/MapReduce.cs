using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace PLinq
{
    static class MapReduceUtil
    {
        public static ParallelQuery<TResult> MapReduce<TSource, TMapped, TKey, TResult>(this
                ParallelQuery<TSource> source,
                Func<TSource, IEnumerable<TMapped>> map,
                Func<TMapped, TKey> keySelector,
                Func<IGrouping<TKey, TMapped>, IEnumerable<TResult>> reduce)
        {
            return source.SelectMany(map)
                    .GroupBy(keySelector)
                    .SelectMany(reduce);
        }



        public static void DuplicatFiles()
        {
            string path = @"c:\temp";

            Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).AsParallel()
                .Select(f =>
                {
                    using (var fs = new FileStream(f, FileMode.Open, FileAccess.Read))
                        return new
                        {
                            FileName = f,
                            FileHash = BitConverter.ToString(SHA1.Create().ComputeHash(fs))
                        };
                })
                .GroupBy(f => f.FileHash, EqualityComparer<string>.Default)
                .Select(g => new { FileHash = g.Key, Files = g.Select(z => z.FileName).ToList() })
                .Where(g => g.Files.Count > 1)
                //.Skip(1).SelectMany(f => f.Files)
                .ForAll(f => Console.WriteLine(f.Files.First()));
        }


        public static void WordCounter()
        {
            char[] delimiters = { ' ', ',', ';', '.' };
            string[] files =
            {
                @"C:\shakespeare\Sonnet 1.txt",
                @"C:\shakespeare\Sonnet 2.txt",
                @"C:\shakespeare\Sonnet 3.txt",
                @"C:\shakespeare\Sonnet 4.txt"
            };

            var counts = files.AsParallel().MapReduce(
                path => File.ReadLines(path)
                    .SelectMany(line => line.Split(delimiters)),
                        word => word,
                        group => new[]
                        {
                            new KeyValuePair<string, int>(group.Key, group.Count())
                        }
            );

            foreach (var word in counts)
            {
                Console.WriteLine(word.Key + " " + word.Value);
            }
            //Console.WriteLine("Press enter to exit");
            Console.ReadLine();
        }
    }
}