#define _CONSOLEAGENT
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;
using Microsoft.FSharp.Control;
using static ParallelizingFuzzyMatch.Data;

namespace DataFlowPipeLine
{
    public class FuzzyMatchDataFlow
    {
        public FuzzyMatchDataFlow(FSharpMailboxProcessor<Tuple<string, string[]>> agent,
            FSharpMailboxProcessor<Tuple<int, string>> consoleAgent, CancellationTokenSource cts)
        {
            this.cts = cts;
            this.agent = agent;
            this.consoleAgent = consoleAgent;
            InputBlock = new BufferBlock<Tuple<string, string>>(
              new DataflowBlockOptions { BoundedCapacity = 5, CancellationToken = cts.Token });
        }

        public CancellationTokenSource cts;
        public FSharpMailboxProcessor<Tuple<string, string[]>> agent;
        public FSharpMailboxProcessor<Tuple<int, string>> consoleAgent;

        public BufferBlock<Tuple<string, string>> InputBlock;

        public async Task ProcessAsynchronously()
        {
            const int MDP = 1;

            var splitLines = new TransformBlock<Tuple<string, string>, Tuple<string, string[]>>(
                n =>
                {
                    string nameText = n.Item1;
                    string text = n.Item2;

                    string[] lines = text.Split('\n');

#if CONSOLEAGENT                    
                    consoleAgent.Post(Tuple.Create(1,string.Format("Text {0} received - Splitting Lines on thread id {1}", nameText, Thread.CurrentThread.ManagedThreadId)));
#else

                    
                    ConsoleColor backupColor = Console.ForegroundColor;
                    Console.ForegroundColor = ConsoleColor.Red;

                    Console.WriteLine("Text {0} received - Splitting Lines on thread id {1}", nameText, Thread.CurrentThread.ManagedThreadId);
                    Console.ForegroundColor = backupColor;
#endif
                    Thread.Sleep(TimeSpan.FromMilliseconds(100));
                    return Tuple.Create(nameText, lines);

                }, new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = MDP, CancellationToken = cts.Token });

            var splitWords = new TransformBlock<Tuple<string, string[]>, Tuple<string, string[]>>(
                n =>
                {
                    string nameText = n.Item1;
                    string[] lines = n.Item2;
#if CONSOLEAGENT
                    consoleAgent.Post(Tuple.Create(2, string.Format("Text {0} received - Splitting Words on thread id {1}", nameText, Thread.CurrentThread.ManagedThreadId)));
#else
                    ConsoleColor backupColor = Console.ForegroundColor;
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("Text {0} received - Splitting Words on thread id {1}", nameText, Thread.CurrentThread.ManagedThreadId);
                    Console.ForegroundColor = backupColor;
#endif
                    string[] words = (from line in lines
                                      from word in line.Split(Delimiters)
                                      select word.ToUpper()).ToArray();

                    Thread.Sleep(TimeSpan.FromMilliseconds(100));
                    return Tuple.Create(nameText, words);
                }, new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = MDP, CancellationToken = cts.Token });

            var fuzzyMatch = new TransformBlock<Tuple<string, string[]>, Tuple<string, string[]>>(
                n =>
                {
                    string nameText = n.Item1;
                    string[] words = n.Item2;

#if CONSOLEAGENT
                    consoleAgent.Post(Tuple.Create(3, string.Format("Text {0} received - Fuzzy Match on thread id {1}", nameText, Thread.CurrentThread.ManagedThreadId)));
#else
                    ConsoleColor backupColor = Console.ForegroundColor;
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine("Text {0} received - Fuzzy Match on thread id {1}", nameText, Thread.CurrentThread.ManagedThreadId);
                    Console.ForegroundColor = backupColor;
#endif

                    var matches = (from wordToSearch in WordsToSearch.AsParallel()
                                   from match in FuzzyMatch.JaroWinklerModule.Parallel.bestMatch(words, wordToSearch)
                                   select match.Word).ToArray();

                    Thread.Sleep(TimeSpan.FromMilliseconds(100));
                    return Tuple.Create(nameText, matches);
                }
                , new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = MDP, CancellationToken = cts.Token });

            var sendBackResult = new ActionBlock<Tuple<string, string[]>>(
                s =>
                {
                    string nameText = s.Item1;
                    string[] matches = s.Item2;

#if CONSOLEAGENT
                  consoleAgent.Post(Tuple.Create(4, string.Format("The final result sending back on thread id {1}", Thread.CurrentThread.ManagedThreadId)));
#else
                    ConsoleColor backupColor = Console.ForegroundColor;
                    Console.ForegroundColor = ConsoleColor.Magenta;
                    Console.WriteLine("The final result sending back on thread id {1}", s, Thread.CurrentThread.ManagedThreadId);
                    Console.ForegroundColor = backupColor;
#endif
                    agent.Post(Tuple.Create(nameText, matches));

                }
                , new ExecutionDataflowBlockOptions { MaxDegreeOfParallelism = MDP, CancellationToken = cts.Token });

            try
            {

                InputBlock.LinkTo(splitLines, new DataflowLinkOptions { PropagateCompletion = true });
                splitLines.LinkTo(splitWords, new DataflowLinkOptions { PropagateCompletion = true });
                splitWords.LinkTo(fuzzyMatch, new DataflowLinkOptions { PropagateCompletion = true });
                fuzzyMatch.LinkTo(sendBackResult, new DataflowLinkOptions { PropagateCompletion = true });

                await sendBackResult.Completion;

                consoleAgent.Post(Tuple.Create(999,
                    string.Format("{0}\nPress ENTER to exit.\n{0}", new string('*', 30))));
            }
            catch (OperationCanceledException)
            {
                Console.WriteLine("Operation has been canceled! Press ENTER to exit.");
            }

            Console.ReadLine();
        }
    }
}