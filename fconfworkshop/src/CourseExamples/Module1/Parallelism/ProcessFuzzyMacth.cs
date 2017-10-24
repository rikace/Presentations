using CommonHelpers;
using Parallelism;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using static Parallelism.Data;

namespace ParallelizingFuzzyMatch
{
    public class ProcessFuzzyMacth
    {

        public void SequentialFuzzyMatch()
        {
            List<string> matches = new List<string>();
            BenchPerformance.Time("Sequential Fuzzy Match", iterations: Data.Iterations, operation: () =>
            {
                foreach (var word in WordsToSearch)
                {
                    var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, word);
                    matches.AddRange(localMathes.Select(m => m.Word));
                }
            });
            foreach (var match in matches.Distinct())
            {
                Console.Write("{0}\t", match);
            }
            Console.WriteLine();
        }

        public void ThreadFuzzyMatch()
        {
            List<string> matches = new List<string>();
            BenchPerformance.Time("Thread Fuzzy Match",
                iterations: Data.Iterations, operation: () =>
                {
                    var t = new Thread(() =>
                    {
                        foreach (var word in WordsToSearch)
                        {
                            var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, word);
                            matches.AddRange(localMathes.Select(m => m.Word));
                        }
                    });
                    t.Start();
                    t.Join();

                });
            foreach (var match in matches.Distinct())
            {
                Console.Write("{0}\t", match);
            }
            Console.WriteLine();
        }

        public void TwoThreadsFuzzyMatch()
        {
            List<string> matches = new List<string>();
            BenchPerformance.Time("Two Thread Fuzzy Match",
                iterations: Data.Iterations, operation: () =>
                {
                    var t1 = new Thread(() =>
                    {
                        var take = WordsToSearch.Count / 2;
                        var start = 0;

                        foreach (var word in WordsToSearch.Take(take))
                        {
                            var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, word);
                            matches.AddRange(localMathes.Select(m => m.Word));
                        }
                    });
                    var t2 = new Thread(() =>
                    {
                        var start = WordsToSearch.Count / 2;
                        var take = WordsToSearch.Count - start;

                        foreach (var word in WordsToSearch.Skip(start).Take(take))
                        {
                            var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, word);
                            matches.AddRange(localMathes.Select(m => m.Word));
                        }
                    });
                    t1.Start();
                    t2.Start();
                    t1.Join();
                    t2.Join();
                });
            foreach (var match in matches.Distinct())
            {
                Console.Write("{0}\t", match);
            }
            Console.WriteLine();
        }

        public void MultipleThreadsFuzzyMatch()
        {
            List<string> matches = new List<string>();
            BenchPerformance.Time("Multi Thread Fuzzy Match",
                iterations: Data.Iterations, operation: () =>
                {
                    var threads = new Thread[Environment.ProcessorCount];

                    for (int i = 0; i < threads.Length; i++)
                    {
                        var index = i;
                        threads[index] = new Thread(() =>
                        {
                            var take = WordsToSearch.Count / (Math.Min(WordsToSearch.Count, threads.Length));
                            var start = index == threads.Length - 1 ? WordsToSearch.Count - take : index * take;
                            foreach (var word in WordsToSearch.Skip(start).Take(take))
                            {
                                var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, word);
                                matches.AddRange(localMathes.Select(m => m.Word));
                            }
                        });
                    }

                    for (int i = 0; i < threads.Length; i++)
                        threads[i].Start();
                    for (int i = 0; i < threads.Length; i++)
                        threads[i].Join();
                });
            foreach (var match in matches.Distinct())
            {
                Console.Write("{0}\t", match);
            }
            Console.WriteLine();
        }

        public void ParallelLoopFuzzyMatch()
        {
            List<string> matches = new List<string>();
            BenchPerformance.Time("Parallel Loop Fuzzy Match",
                iterations: Data.Iterations, operation: () =>
                {
                    object sync = new object();

                    Parallel.ForEach(WordsToSearch,
                                        // thread local initializer
                                        () => { return new List<string>(); },
                                        (word, loopState, localMatches) =>
                                        {
                                            var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, word);
                                            localMatches.AddRange(localMathes.Select(m => m.Word));// same code 
                                        return localMatches;
                                        },
                        (finalResult) =>
                        {
                        // thread local aggregator
                        lock (sync) matches.AddRange(finalResult);
                        }
                    );
                });
            foreach (var match in matches.Distinct())
            {
                Console.Write("{0}\t", match);
            }
            Console.WriteLine();
        }

        public void MultipleTasksFuzzyMatch()
        {
            var tasks = new List<Task<List<string>>>();
            var matches = new List<string>();
            BenchPerformance.Time("Multi Tasks Fuzzy Match",
                iterations: Data.Iterations, operation: () =>
                {
                    foreach (var word in WordsToSearch)
                    {
                        tasks.Add(Task.Factory.StartNew<List<string>>((w) =>
                        {
                            List<string> localMatches = new List<string>();
                            var localMathes = FuzzyMatch.JaroWinklerModule.bestMatch(Words, (string)w);
                            localMatches.AddRange(localMathes.Select(m => m.Word));
                            return localMatches;
                        }, word));
                    }

                    Task.Factory.ContinueWhenAll(tasks.ToArray(), (ts) =>
                    {
                        matches = new List<string>(tasks.SelectMany(t => t.Result).Distinct());
                    }).Wait();
                });
            foreach (var match in matches)
            {
                Console.Write("{0}\t", match);
            }
            Console.WriteLine();
        }

        public void LinqFuzzyMatch()
        {
            BenchPerformance.Time("Linq Fuzzy Match", () =>
            {
                var matches = (from word in WordsToSearch
                               from match in FuzzyMatch.JaroWinklerModule.bestMatch(Words, word)
                               select match.Word);

                foreach (var match in matches)
                {
                    Console.Write("{0}\t\t", match);
                }

                Console.WriteLine();
            });
        }

        public void ParallelLinqFuzzyMatch()
        {
            BenchPerformance.Time("Parallel Linq Fuzzy Match",
                iterations: Data.Iterations, operation: () =>
                {
                    ParallelQuery<string> matches = (from word in WordsToSearch.AsParallel()
                                                     from match in FuzzyMatch.JaroWinklerModule.bestMatch(Words, word)
                                                     select match.Word);

                    foreach (var match in matches)
                    {
                        Console.Write("{0}\t", match);
                    }

                    Console.WriteLine();
                });
        }

        public void ParallelArrayFuzzyMatch()
        {
            BenchPerformance.Time("Parallel Array F# Fuzzy Match", () =>
            {
                var matches = (from word in WordsToSearch
                               from match in FuzzyMatch.JaroWinklerModule.Parallel.bestMatch(Words, word)
                               select match.Word);

                foreach (var match in matches)
                {
                    Console.Write("{0}\t", match);
                }

                Console.WriteLine();
            });
        }
    }
}
