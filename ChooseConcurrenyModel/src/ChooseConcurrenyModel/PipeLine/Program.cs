using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PipeLine
{
    public class Program
    {
        public static void Main()
        {
            var bufferA = new BlockingCollection<int>(20);
            var bufferB = new BlockingCollection<int>(20);

            var createStage = Task.Factory.StartNew(() =>
            {
                CreateRange(bufferA);
            }, TaskCreationOptions.LongRunning);

            var squareStage = Task.Factory.StartNew(() =>
            {
                SquareTheRange(bufferA, bufferB);
            }, TaskCreationOptions.LongRunning);

            var displayStage = Task.Factory.StartNew(() =>
            {
                DisplayResults(bufferB);
            }, TaskCreationOptions.LongRunning);

            Task.WaitAll(createStage, squareStage, displayStage);

            Console.ReadLine();
        }

        static void CreateRange(BlockingCollection<int> result)
        {
            try
            {
                for (int i = 1; i < 20; i++)
                {
                    result.Add(i);
                    Console.WriteLine("Create Range {0}", i);
                }
            }
            finally
            {
                result.CompleteAdding();
            }
        }

        static void SquareTheRange(BlockingCollection<int> source, BlockingCollection<int> result)
        {
            try
            {
                foreach (var value in source.GetConsumingEnumerable())
                {
                    result.Add((int)(value * value));
                }
            }
            finally
            {
                result.CompleteAdding();
            }
        }

        static void DisplayResults(BlockingCollection<int> input)
        {
            foreach (var value in input.GetConsumingEnumerable())
            {
                Console.WriteLine("The result is {0}", value);
            }
        }


        public static void FilteringPipeline()
        {
            //Generate the source data.
            var source = new BlockingCollection<int>[3];
            for (int i = 0; i < source.Length; i++)
                source[i] = new BlockingCollection<int>(100);

            Parallel.For(0, source.Length * 100, (data) =>
            {
                int item = BlockingCollection<int>.TryAddToAny(source, data);
                if (item >= 0)
                    Console.WriteLine("added {0} to source data", data);
            });

            foreach (var array in source)
                array.CompleteAdding();

            // calculate the square 
            var calculateFilter = new PipelineFilter<int, int>
            (
                source,
                (n) => n * n,
                "calculateFilter"
             );

            //Convert ints to strings
            var convertFilter = new PipelineFilter<int, string>
            (
                calculateFilter.m_outputData,
                (s) => String.Format("{0}", s),
                "convertFilter"
             );

            // Displays the results
            var displayFilter = new PipelineFilter<string, string>
            (
                convertFilter.m_outputData,
                (s) => Console.WriteLine("The final result is {0}", s),
                "displayFilter");

            // Start the pipeline
            try
            {
                Parallel.Invoke(
                             () => calculateFilter.Run(),
                             () => convertFilter.Run(),
                             () => displayFilter.Run()
                         );
            }
            catch (AggregateException aggregate)
            {
                foreach (var exception in aggregate.InnerExceptions)
                    Console.WriteLine(exception.Message + exception.StackTrace);
            }

            Console.ReadLine();
        }
    }


    class PipelineFilter<TInput, TOutput>
    {
        Func<TInput, TOutput> m_function = null;
        public BlockingCollection<TInput>[] m_inputData = null;
        public BlockingCollection<TOutput>[] m_outputData = null;
        Action<TInput> m_outputAction = null;
        public string Name { get; private set; }

        public PipelineFilter(BlockingCollection<TInput>[] input, Func<TInput, TOutput> processor, string name)
        {
            m_inputData = input;
            m_outputData = new BlockingCollection<TOutput>[3];
            for (int i = 0; i < m_outputData.Length; i++)
                m_outputData[i] = new BlockingCollection<TOutput>(100);

            m_function = processor;
            Name = name;
        }

        //used for final endpoint 
        public PipelineFilter(BlockingCollection<TInput>[] input, Action<TInput> renderer, string name)
        {
            m_inputData = input;
            m_outputAction = renderer;
            Name = name;
        }

        public void Run()
        {
            Console.WriteLine("filter {0} is running", this.Name);
            while (!m_inputData.All(bc => bc.IsCompleted))
            {
                TInput receivedItem;
                int i = BlockingCollection<TInput>.TryTakeFromAny(
                    m_inputData, out receivedItem, 50);
                if (i >= 0)
                {
                    if (m_outputData != null)
                    {
                        TOutput outputItem = m_function(receivedItem);
                        BlockingCollection<TOutput>.AddToAny(m_outputData, outputItem);
                        Console.WriteLine("{0} sent {1} to next filter", this.Name, outputItem);
                    }
                    else
                    {
                        m_outputAction(receivedItem);
                    }
                }
                else
                    Console.WriteLine("Could not get data from previous filter");
            }
            if (m_outputData != null)
            {
                foreach (var bc in m_outputData) bc.CompleteAdding();
            }
        }
    }
}