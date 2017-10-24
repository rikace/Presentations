using ParallelCompressionCS;
using ReactiveAgent.Agents;
using System;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using ReactiveAgent.Agents.Dataflow;
using System.Collections.Generic;
using FSharp.Charting;
using Microsoft.FSharp.Core;
using System.Windows.Forms;
using System.Threading.Tasks.Dataflow;

namespace ReactiveAgent.CS
{
    public class Program
    {
        private static async Task ReadDataAsync(BufferBlock<int> bufferingBlock)
        {
            // Receive the messages back . 
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(await bufferingBlock.ReceiveAsync());
            }
        }

        private static async Task WriteDataAsync(BufferBlock<int> bufferingBlock)
        {
            // Post some messages to the block. 
            for (int i = 0; i < 10; i++)
            {
                await bufferingBlock.SendAsync(i * i);
            }
        }


        public static void Main(string[] args)
        {
            // Create a BufferBlock object. 
            var bufferingBlock = new BufferBlock<int>();

            WriteDataAsync(bufferingBlock).Wait();
            ReadDataAsync(bufferingBlock).Wait();

            Console.WriteLine("Finished. Press any key to exit.");
            Console.ReadLine();


            //Play().Wait();
            RunPerfComparison(new[] { 3, 6, 9 }, new[] { 1, 2, 4 });
            return;

            // 6 GB
            //CreateTextFileWithSize(sourceFile_6000, sourceFile_132, 6L * 1024 * 1024 * 1024);

            // 12 Gb
            CreateTextFileWithSize(sourceFile_Generated, sourceFile_132, 12L * 1024 * 1024 * 1024);


            // 12.8, 12.9
            CompressAndEncrypt(sourceFile_Generated, destinationFile, restoredFile);
        }

        static async Task Play()
        {
            (new DataflowTransformActionBlocks()).Run();
           
            (new StatefulDataflowAgentSample()).Run();

            await (new WordCountAgentsExample()).Run();
        }

        private static string workDirectory = @"Data";
        private static string sourceFile_132 = @"text.txt";
        private static string sourceFile_Generated = @"testText.txt";
        private static string destinationFile = @"text.zip";
        private static string restoredFile = @"text_restored.txt";

        public static void CompressAndEncrypt(string srcFile, string dstFile, string rstFile)
        {
            try
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();

                Console.WriteLine("CompressAndEncrypt ...");
                using (var streamSource = new FileStream(srcFile, FileMode.OpenOrCreate, FileAccess.Read, FileShare.None, 0x1000, useAsync: true))
                using (var streamDestination = new FileStream(dstFile, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None, 0x1000, useAsync: true))
                {
                    CompressionAndEncryptDataFlow.CompressAndEncrypt(streamSource, streamDestination).Wait();
                    streamDestination.Close();
                }
                Console.WriteLine($"Done in {sw.ElapsedMilliseconds}");

                Console.WriteLine("Press Enter to continue");
                Console.ReadLine();
                sw.Restart();

                Console.WriteLine("DecryptAndDecompress ...");
                using (var streamSource = new FileStream(dstFile, FileMode.OpenOrCreate, FileAccess.Read, FileShare.None, 0x1000, useAsync: true))
                using (var streamDestination = new FileStream(rstFile, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None, 0x1000, useAsync: true))
                    CompressionAndEncryptDataFlow.DecryptAndDecompress(streamSource, streamDestination).Wait();
                Console.WriteLine($"Done in {sw.ElapsedMilliseconds}");

                Console.WriteLine("Press Enter to continue");
                Console.ReadLine();

                Console.WriteLine("Verification ...");
                using (var f1 = System.IO.File.OpenRead(srcFile))
                using (var f2 = System.IO.File.OpenRead(rstFile))
                {
                    bool ok = false;
                    if (f1.Length == f2.Length)
                    {
                        ok = true;
                        int count;
                        const int size = 0x1000000;

                        var buffer = new byte[size];
                        var buffer2 = new byte[size];

                        while ((count = f1.Read(buffer, 0, buffer.Length)) > 0 && ok == true)
                        {
                            f2.Read(buffer2, 0, count);
                            ok = buffer2.SequenceEqual(buffer);
                            if (!ok) break;
                        }
                    }
                    Console.WriteLine($"Restored file isCorrect = {ok}");
                }

                Console.WriteLine("Press Enter to continue");
                Console.ReadLine();
            }
            catch (AggregateException ex)
            {
                var q = new Queue<Exception>(new[] { ex });
                while (q.Count > 0)
                {
                    var e = q.Dequeue();
                    Console.WriteLine($"\t{e.Message}");
                    if (e is AggregateException)
                    {
                        foreach (var x in (e as AggregateException).InnerExceptions)
                            q.Enqueue(x);
                    }
                    else
                    {
                        if (e.InnerException != null)
                            q.Enqueue(e.InnerException);
                    }
                }
            }
        }

        public static void CreateTextFileWithSize(string path, string templateFilePath, long targetSize) {
            var bytes = System.IO.File.ReadAllBytes(templateFilePath);

            using (FileStream fs = new FileStream(path, FileMode.Append, FileAccess.Write))
            {
                var iterations = (targetSize - fs.Length + bytes.Length) / bytes.Length;

                for (var i = 0; i < iterations; i++)
                    fs.Write(bytes, 0, bytes.Length);
            }
        }

        public static void RunPerfComparison(int[] fileSizesInGb, int[] degreesOfParallelism)
        {
            const long bytesInGb = 1024L * 1024 * 1024;
            string templateFile = sourceFile_132;

            if (!Directory.Exists(workDirectory))
                Directory.CreateDirectory(workDirectory);

            string inFile = Path.Combine(workDirectory, "inFile.txt");
            string outFile = Path.Combine(workDirectory, "outFile.txt");

            var results = new List<List<TimeSpan>>();

            foreach (var size in fileSizesInGb)
            {
                Console.WriteLine($"Creating input file {size}GB ...");
                if (System.IO.File.Exists(inFile))
                    System.IO.File.Delete(inFile);
                CreateTextFileWithSize(inFile, templateFile, bytesInGb * size);

                for(var i=0; i<degreesOfParallelism.Length; i++)
                {
                    var dop = degreesOfParallelism[i];
                    if (System.IO.File.Exists(outFile))
                        System.IO.File.Delete(outFile);

                    Console.WriteLine($"Running compression with degreeOfParallelism={dop} ...");
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    RunCompression(inFile, outFile, dop);
                    sw.Stop();
                    Console.WriteLine($"\t Elapsed = {sw.Elapsed}");

                    if (i == results.Count)
                        results.Add(new List<TimeSpan>());
                    results[i].Add(sw.Elapsed);
                }
            }

            var keys = fileSizesInGb.Select(s => $"{s}Gb").ToList();
            var charts = new List<ChartTypes.GenericChart>();
            for (var i=0; i<results.Count; i++)
            {
                var line = results[i].Select(t => t.TotalMilliseconds).ToList();
                var lineChart = PerfVis.CreateLineChart(line, keys, $"dop={degreesOfParallelism[i]}");
                charts.Add(lineChart);
            }
            Application.Run(PerfVis.CombineToForm(charts, "Dataflow compression with different degree of parallelism(dop)"));
        }

        public static void RunCompression(string srcFile, string dstFile, int degreeOfParallelism)
        {
            using (var streamSource = new FileStream(srcFile, FileMode.OpenOrCreate, FileAccess.Read, FileShare.None, 0x1000, useAsync: true))
            using (var streamDestination = new FileStream(dstFile, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None, 0x1000, useAsync: true))
            {
                CompressionAndEncryptDataFlow.CompressAndEncrypt(streamSource, streamDestination, degreeOfParallelism).Wait();
                streamDestination.Close();
            }
        }
    }
}