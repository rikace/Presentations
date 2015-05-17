using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using EventNet;

namespace Console
{
    class Program
    {
        public void FireFSharpEvent()
        {
            EventNet.Event.MyEvent myEvent = new Event.MyEvent();
            myEvent.Event += myEvent_Event;
            myEvent.Raise(EventArgs.Empty);
        }

        void myEvent_Event(object sender, EventArgs value)
        {
            System.Console.WriteLine("event fired!!");
        }

        static void Main(string[] args)
        {
            Program p = new Program();
            p.FireFSharpEvent();

            RunTask();

            System.Console.ReadLine();
        }

        private static void RunTask()
        {
            var task = AsyncTask.Task.getData("http://www.fsharp.org");

            task.ContinueWith(lines =>
            {
                foreach (var line in task.Result)
                    System.Console.WriteLine(line);
            });
        }
    }
}
