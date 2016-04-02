using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using static ParallelizingFuzzyMatch.Data;

namespace ParallelizingFuzzyMatch
{
    public class AsyncLoadFiles
    {
        public IEnumerable<string> LoadDataSequential()
        {
            var urls = GetTextFileUrls();
            List<string> data = new List<string>();
            foreach (var url in urls)
            {
                using (var client = new WebClient())
                {
                    const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                    client.Headers.Add("user-agent", headerText);
                    string text = client.DownloadString(url.Value);
                    data.Add(text);
                }
            }
            return data;
        }

        public IEnumerable<string> LoadDataMultiTasks()
        {
            var urls = GetTextFileUrls();
            List<string> data = new List<string>();
            var tasks = new List<Task<string>>();

            foreach (var url in urls)
            {
                tasks.Add(Task.Factory.StartNew<string>((u) =>
                {
                    using (var client = new WebClient())
                    {
                        const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                        client.Headers.Add("user-agent", headerText);
                        string text = client.DownloadString(url.Value);
                        return text;
                    }
                }, url));
            }

            Task.Factory.ContinueWhenAll(tasks.ToArray(), (ts) =>
            {
                data.AddRange(tasks.Select(t => t.Result));
            }).Wait();

            return data;
        }


        public async Task<IEnumerable<string>> LoadDataAsync()
        {
            var urls = GetTextFileUrls();
            List<string> data = new List<string>();

            foreach (var url in urls)
            {
                using (var client = new WebClient())
                {
                    const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                    client.Headers.Add("user-agent", headerText);
                    string text = await client.DownloadStringTaskAsync(url.Value);
                    data.Add(text);
                }
            }
            return data;
        }
        public async Task<IEnumerable<string>> LoadDataMultipleAsync()
        {
            var urls = GetTextFileUrls();
            List<string> data = new List<string>();
            var tasks = new List<Task<string>>();
            foreach (var url in urls)
            {
                using (var client = new WebClient())
                {
                    const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                    client.Headers.Add("user-agent", headerText);
                    Task<string> task = client.DownloadStringTaskAsync(url.Value);
                    tasks.Add(task);
                }
            }
            foreach (var task in tasks)
            {
                string text = await task;
                data.Add(text);
            }
            return data;
        }

        async Task<string> DownloadData(string url)
        {
            using (var client = new WebClient())
            {
                const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                client.Headers.Add("user-agent", headerText);
                Task<string> task = client.DownloadStringTaskAsync(url);
                return await task;
            }
        }

        public async Task<IEnumerable<string>> LoadDataMultipleParallelAsync()
        {
            var urls = GetTextFileUrls();
            List<string> data = new List<string>();

            var client = new WebClient();
            try
            {
                // Create a query that, when executed, returns a collection of tasks.
                IEnumerable<Task<string>> tasks = from url in urls
                                                  select DownloadData(url.Value);

                // Use ToList to execute the query and start the tasks. 
                List<Task<string>> downloadTasks = tasks.ToList();

                // Add a loop to process the tasks one at a time until none remain.
                while (downloadTasks.Count > 0)
                {
                    // Identify the first task that completes.
                    Task<string> firstFinishedTask = await Task.WhenAny(downloadTasks);

                    // Remove the selected task from the list so that you don't
                    // process it more than once.
                    downloadTasks.Remove(firstFinishedTask);

                    // Await the completed task.
                    string text = await firstFinishedTask;
                    data.Add(text);
                }
                return data;
            }
            finally
            {
                client.Dispose();
            }
        }


        public async Task<IEnumerable<string>> LoadDataMultipleParallelAndProcessAsync()
        {
            var urls = GetTextFileUrls();
            List<Task<List<string>>> matchTasks = new List<Task<List<string>>>();

            var client = new WebClient();
            try
            {
                // Create a query that, when executed, returns a collection of tasks.
                IEnumerable<Task<string>> tasks = from url in urls
                                                  select DownloadData(url.Value);

                // Use ToList to execute the query and start the tasks. 
                List<Task<string>> downloadTasks = tasks.ToList();

                // Add a loop to process the tasks one at a time until none remain.
                while (downloadTasks.Count > 0)
                {
                    // Identify the first task that completes.
                    Task<string> firstFinishedTask = await Task.WhenAny(downloadTasks);

                    // Remove the selected task from the list so that you don't
                    // process it more than once.
                    downloadTasks.Remove(firstFinishedTask);

                    // Await the completed task.
                    string text = await firstFinishedTask;

                    var words =
                        from lines in text.Split('\n')
                        from word in lines.Split(Delimiters)
                        select word.ToUpper();

                    var task = Task.Run<List<string>>(() =>
                    {
                        return (from wordToSearch in WordsToSearch
                                from match in FuzzyMatch.JaroWinklerModule.Parallel.bestMatch(words.ToArray(), wordToSearch)
                                select match.Word).ToList();
                    });
                    matchTasks.Add(task);
                }
                await Task.WhenAll(matchTasks);
                return matchTasks.SelectMany(t => t.Result);
            }
            finally
            {
                client.Dispose();
            }
        }
    }
}
