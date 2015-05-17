using FCSlib;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSMisc
{
    internal class CurrySamples
    {
        public static void Currying()
        {
            Func<int, int, int> add = delegate(int x, int y)
            {
                return x + y;
            };

            Func<int, Func<int, int>> curriedAdd = delegate(int x)
            {
                return delegate(int y)
                {
                    return x + y;
                };
            };

            Func<int, int, int> addL = (x, y) => x + y;
            Func<int, Func<int, int>> curriedAddL = x => y => x + y;

            int result_is_5 = add(3, 2);
            int result_is_6 = curriedAdd(2)(4);

            var add2 = curriedAdd(2);

            int result_is_7 = add2(5);


            var addCurry = Functional.Curry<int, int, int>(add);
            var add3 = addCurry(3);
            var result_is_9 = add3(6);
        }

        public static void test()
        {

            Func<Func<int, int>, int, int, int> strangeSum = null;
            strangeSum = (f, x, y) => x > y ? 0 : f(x) + strangeSum(f, (x + 1), y);

            Func<int, int> square = (x) => x * x;
            Func<int, int, int> multiplier = (a, b) => a * b;
            Func<int, int> fact = null;
            fact = (x) => x == 0 ? 1 : x * fact(x - 1);

            Func<int, int, int> sumSquares = (x, y) => strangeSum(square, x, y);
            Func<int, int, int> sumFactorials = (x, y) => strangeSum(fact, x, y);
            var sumFactorialsResult = sumFactorials(4, 4);
            var sumSqauresResult = sumSquares(2, 3);






            Func<int, int, Func<int>> sumSquaresLazy = (x, y) => () => strangeSum(square, x, y);
            Func<int, int, Func<int>> sumFactorialsLazy = (x, y) => () => strangeSum(fact, x, y);



            Console.WriteLine(sumFactorialsResult);
            Console.WriteLine(sumSqauresResult);



            int value = Multiply()(2)(3);
            Console.WriteLine(value);

            var doubleFunction = Multiply()(2);
            Console.WriteLine((int)doubleFunction(3));
            Console.WriteLine((int)doubleFunction(5));


            /*
             // multiply' :: int * int -> int
             let multiply'(x1,x2) = x1 * x2
              
             // multiply :: int -> int -> int
let multiply x1 x2 = x1 * x2
              
             let value = multiply 2 3
printfn "%d" value
              
             // double :: int -> int
let double x1 = multiply 2 x1
 
// triple :: int -> int
let triple x1 = multiply 3 x1
 
let value' = double 5
printfn "%d" value'
             * 
             * 
             */
        }


        public static int Multiply(int x1, int x2)
        {
            return x1 * x2;
        }

        public static int Double(int x1)
        {
            return Multiply(2, x1);
        }


        public static Func<int, Func<int, int>> Multiply()
        {
            return x1 => (x2 => x1 * x2);
        }

        public static Func<int, int> Double()
        {
            return Multiply()(2);
        }


        //public void test2()
        //{

        //    int value = Multiply()(2)(3);

        //    var doubleFunction = Multiply()(2);
        //    var result1 = doubleFunction(3);
        //    var result2 = doubleFunction(5);





        //    Func<int, Func<int, int>> Multiply = x => y => x * y;
        //    Func<int, int> MultiplyBy2 = x => Multiply(x)(2);

        //    int result_Is_10 = Multiply(2)(5);
        //    int result_Is_12 = MultiplyBy2(6);


    }
}