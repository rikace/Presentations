using System;
using System.Threading.Tasks.Dataflow;
using System.Threading.Tasks;

namespace DataflowReadWriteAsync
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a BufferBlock object. 
            var bufferingBlock = new BufferBlock<int>();

            WriteDataAsync(bufferingBlock).Wait();
            ReadDataAsync(bufferingBlock).Wait();

            Console.WriteLine("Finished. Press any key to exit.");
            Console.ReadLine();
        }

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
    }
}
