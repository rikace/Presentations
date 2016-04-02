using System;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace ProducerConsumer
{
    class Program
    {
        static void Main(string[] args)
        {
            
            var buffer = new BufferBlock<int>();

            // Start the consumer.   
            var consumer = ConsumeAsync(buffer);

            // Post source data.
            Produce(buffer);

            // Wait for the consumer to process data.
            consumer.Wait();

            // Print the count of bytes processed to the console.
            Console.WriteLine("Sum of processed numbers: {0}.", consumer.Result);

            Console.WriteLine("Finished. Press any key to exit.");
            Console.ReadLine();
        }

        static void Produce(ITargetBlock<int> target)
        {
            // Create a Random object.
            Random rand = new Random();

            // fill a buffer with random data  
            for (int i = 0; i < 100; i++)
            {
                // get the next random number 
                int number = rand.Next();

                // Post the result .
                target.Post(number);
            }

            // Set the target to the completed state
            target.Complete();
        }

        static async Task<int> ConsumeAsync(ISourceBlock<int> source)
        {
            // Initialize a counter to track the sum. 
            int sumOfProcessed = 0;

            // Read from the source buffer until empty
            while (await source.OutputAvailableAsync())
            {
                int data = source.Receive();

                // calculate the sum.
                sumOfProcessed += data;
            }

            return sumOfProcessed;
        }

    }
}
