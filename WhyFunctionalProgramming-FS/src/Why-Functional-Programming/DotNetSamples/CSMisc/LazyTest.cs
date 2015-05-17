using System;

namespace CSMisc
{
    public class LazyTest
    {
        public int GetValueX()
        {
            int x = 4;
            Console.WriteLine("Getting value x:{0}", x);
            return x;
        }

        public int GetValueY()
        {
            int y = 5;
            Console.WriteLine("Getting value y:{0}", y);
            return y;
        }

        public int AddIfTrue(bool calculate, int x, int y)
        {
            Console.WriteLine("Calculate is {0}", calculate);
            if (calculate)
            {
                int result = x + y;
                Console.WriteLine("Add x:{0} y:{1} = {2}", x, y, result);
                return result;
            }
            return 0;
        }

        public int AddIfTrueLazy(bool calculate, Func<int> x, Func<int> y)
        {
            Console.WriteLine("Calculate is {0}", calculate);
            if (calculate)
            {
                var xv = x();
                var yv = y();
                int result = xv + yv;
                Console.WriteLine("Add x:{0} y:{1} = {2}", xv, yv, result);
                return xv + yv;
            }
            return 0;
        }
    }
}

