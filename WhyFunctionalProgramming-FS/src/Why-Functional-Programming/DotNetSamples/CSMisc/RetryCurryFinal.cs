using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;

namespace CSMisc.RetryCurryFinal
{

    public static class Extensions
    {
        public static T WithRetry<T>(this Func<T> action)
        {
            var result = default(T);
            int retryCount = 0;

            bool succesful = false;
            do
            {
                try
                {
                    result = action();
                    succesful = true;
                }
                catch (Exception ex)
                {
                    retryCount++;
                }
            } while (retryCount < 3 && !succesful);

            return result;
        }
    }

    public class RetryMethodTest
    {
        public static void ExecuteFileTest()
        {
            // retry is an higerh order function because it takes a function is a parametr
            var filePath = "TextFile.txt";
            Func<string> read = () => System.IO.File.ReadAllText(filePath);
            var text = read.WithRetry();

            Console.WriteLine("Length text {0}", text.Length);
            // what's happend if I have a function that take a parameter ??

            Func<string, string> read2 = (path) => System.IO.File.ReadAllText(path);

            Func<Func<string, string>, string, Func<string>> partial = (f, s) => () => f(s);

            string text2 = read2.Partial(filePath).WithRetry();
            Console.WriteLine("Length text2 {0}", text2.Length);
        }
        public void Execute()
        {
            // retry is an higerh order function because it takes a function is a parametr
            var msft = "http://microsoft.com";
            var client = new WebClient();
            Func<string> download = () => client.DownloadString(msft);
            var @string = download.WithRetry();

            // what's happend if I have a function that take a parameter ??

            // one solution is to write a new Retry method with 
            // the extra parameter, 
            // or using a Functional approach we can adapt 
            // the incoming function with partial 
            // function application 
            Func<string, string> download2 = url => client.DownloadString(url);
            download2.Partial(msft).WithRetry();

            // with curry we can transform a function that takes n parameters into 
            // a function that you invoke to apply a parameter and get back a function
            // that takes n-1 parameters
            Func<string, Func<string>> dCurry = download2.Curry();
            var data = dCurry(msft).WithRetry();
        }

        public string print(string name, int age, DateTime dob) /*(...)*/
        {
            Console.WriteLine(name);
            Console.WriteLine(age);
            Console.WriteLine(dob.ToShortDateString());
            return string.Format("Name {0} - Age {1}", name, age);
        }

        public Func<string, Func<int, Func<DateTime, string>>>
                curry(Func<string, int, DateTime, string> f)
        {
            return (name) => (age) => (dob) => f(name, age, dob);
        }
        public void TestPrintCurry()
        {
            var curriedPrint = curry(print);
            curriedPrint("Ricky")(39)(new DateTime(1975, 05, 10));

            Func<int, Func<DateTime, string>> curriedPrint2 = curriedPrint("Bryony");
            Func<DateTime, string> curriedPrint3 = curriedPrint2(39);
            string result = curriedPrint3(new DateTime(1975, 05, 10));
        }
    }
}


