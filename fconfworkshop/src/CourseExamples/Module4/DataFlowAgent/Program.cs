using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace DataFlowAgent
{
    class Program
    {
        static void Main(string[] args)
        {
            List<string> urls = new List<string> {
                @"http://www.google.com",
                @"http://www.microsoft.com",
                @"http://www.bing.com",
                @"http://www.google.com"
            };
            var agentStateful = new StatefulDataflowAgent<ImmutableDictionary<string, string>, string>(ImmutableDictionary<string, string>.Empty,
               async (ImmutableDictionary<string, string> state, string url) => {  
       if (!state.TryGetValue(url, out string content))
                       using (var webClient = new WebClient())
                       {
                           content = await webClient.DownloadStringTaskAsync(url);
                           Console.WriteLine(content);
                           return state.Add(url, content);    
           }
                   return state;        
   });
            urls.ForEach(url => agentStateful.Post(url));

        }
    }
}
