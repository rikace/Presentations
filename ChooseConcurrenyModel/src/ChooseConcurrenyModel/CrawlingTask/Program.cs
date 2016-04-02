using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace CrawlingTask
{
    class Program
    {
        static void Main(string[] args)
        {
            CreateLinks();
            Task t = RunProgram();
            t.Wait();
        }

        static Dictionary<string, string[]> _contentEmulation = new Dictionary<string, string[]>();

        static async Task RunProgram()
        {
            var bag = new ConcurrentBag<CrawlingTask>();

            string[] urls = new[] { "http://microsoft.com/", "http://google.com/", "http://facebook.com/", "http://twitter.com/" };

            var crawlers = new Task[4];
            for (int i = 1; i <= 4; i++)
            {
                string crawlerName = "Crawler " + i.ToString();
                bag.Add(new CrawlingTask { UrlToCrawl = urls[i - 1], ProducerName = "root" });
                crawlers[i - 1] = Task.Run(() => Crawl(bag, crawlerName));
            }

            await Task.WhenAll(crawlers);
        }

        static async Task Crawl(ConcurrentBag<CrawlingTask> bag, string crawlerName)
        {
            CrawlingTask task;
            while (bag.TryTake(out task))
            {
                IEnumerable<string> urls = await GetLinksFromContent(task);
                if (urls != null)
                {
                    foreach (var url in urls)
                    {
                        var t = new CrawlingTask
                        {
                            UrlToCrawl = url,
                            ProducerName = crawlerName
                        };

                        bag.Add(t);
                    }
                }
                Console.WriteLine("Indexing url {0} posted by {1} is completed by {2}!",
                    task.UrlToCrawl, task.ProducerName, crawlerName);
            }
        }

        static async Task<IEnumerable<string>> GetLinksFromContent(CrawlingTask task)
        {
            await GetRandomDelay();

            if (_contentEmulation.ContainsKey(task.UrlToCrawl)) return _contentEmulation[task.UrlToCrawl];

            return null;
        }

        static void CreateLinks()
        {
            _contentEmulation["http://microsoft.com/"] = new[] { "http://microsoft.com/a.html", "http://microsoft.com/b.html" };
            _contentEmulation["http://microsoft.com/a.html"] = new[] { "http://microsoft.com/c.html", "http://microsoft.com/d.html" };
            _contentEmulation["http://microsoft.com/b.html"] = new[] { "http://microsoft.com/e.html" };

            _contentEmulation["http://google.com/"] = new[] { "http://google.com/a.html", "http://google.com/b.html" };
            _contentEmulation["http://google.com/a.html"] = new[] { "http://google.com/c.html", "http://google.com/d.html" };
            _contentEmulation["http://google.com/b.html"] = new[] { "http://google.com/e.html", "http://google.com/f.html" };
            _contentEmulation["http://google.com/c.html"] = new[] { "http://google.com/h.html", "http://google.com/i.html" };

            _contentEmulation["http://facebook.com/"] = new[] { "http://facebook.com/a.html", "http://facebook.com/b.html" };
            _contentEmulation["http://facebook.com/a.html"] = new[] { "http://facebook.com/c.html", "http://facebook.com/d.html" };
            _contentEmulation["http://facebook.com/b.html"] = new[] { "http://facebook.com/e.html" };

            _contentEmulation["http://twitter.com/"] = new[] { "http://twitter.com/a.html", "http://twitter.com/b.html" };
            _contentEmulation["http://twitter.com/a.html"] = new[] { "http://twitter.com/c.html", "http://twitter.com/d.html" };
            _contentEmulation["http://twitter.com/b.html"] = new[] { "http://twitter.com/e.html" };
            _contentEmulation["http://twitter.com/c.html"] = new[] { "http://twitter.com/f.html", "http://twitter.com/g.html" };
            _contentEmulation["http://twitter.com/d.html"] = new[] { "http://twitter.com/h.html" };
            _contentEmulation["http://twitter.com/e.html"] = new[] { "http://twitter.com/i.html" };
        }

        static Task GetRandomDelay()
        {
            int delay = new Random(DateTime.Now.Millisecond).Next(150, 200);
            return Task.Delay(delay);
        }

        class CrawlingTask
        {
            public string UrlToCrawl { get; set; }

            public string ProducerName { get; set; }
        }
    }
}