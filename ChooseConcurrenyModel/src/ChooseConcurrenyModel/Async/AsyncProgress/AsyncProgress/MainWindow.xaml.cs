using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;

namespace AsyncProgress
{
    public partial class MainWindow : Window
    {
        char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
        const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
        CancellationTokenSource cts;
        long totalBytes = 0;
        long totalBytesRead = 0;

        public MainWindow()
        {
            InitializeComponent();
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            totalBytes = 0;
            totalBytesRead = 0; DownloadProgress.Value = totalBytesRead;
            TextResult.Text = "";
            TextResult.Text += "Started downloading...\n";
            cts = new CancellationTokenSource();
            //Task<int> countTask = GetWordCountAsync();
            //int result = await countTask;
            await GetMultipleWordCount();
            TextResult.Text += String.Format("Finished downloading.\n");
        }
        
        async Task<KeyValuePair<string, int>> ProcessBook2(KeyValuePair<string, string> book, CancellationToken ct)
        {
            await Task.Delay(500);
            var client = new WebClient();
            try
            {
                TextResult.Text += String.Format("Getting the word count for {0}...\n", book.Key);
                client.Headers.Add("user-agent", headerText);

                using (var stream = await client.OpenReadTaskAsync(book.Value))
                    Interlocked.Add(ref totalBytes, Convert.ToInt64(client.ResponseHeaders["Content-Length"]));

                client.DownloadProgressChanged += new DownloadProgressChangedEventHandler(client_DownloadProgressChanged);
                client.DownloadFileCompleted += new AsyncCompletedEventHandler(client_DownloadFileCompleted);
                Task<string> wordsTask = client.DownloadStringTaskAsync(new Uri(book.Value));
                var words = await wordsTask;
                var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                return new KeyValuePair<string, int>(book.Key, wordArray.Count());
            }
            catch (OperationCanceledException)
            {
                TextResult.Text += "Download cancelled.\n";
            }
            catch (Exception ex)
            {
                TextResult.Text += String.Format("An error has occurred: {0} \n", ex.Message);
            }
            finally
            {
                client.Dispose();
            }
            
            return new KeyValuePair<string, int>("", 0);
        }


        void client_DownloadProgressChanged(object sender, DownloadProgressChangedEventArgs e)
        {
            long bytesIn = long.Parse(e.BytesReceived.ToString());
            Interlocked.Add(ref totalBytesRead, bytesIn);
            double percentage = totalBytesRead / totalBytes * 100;
            DownloadProgress.Value = int.Parse(Math.Truncate(percentage).ToString());
        }
        void client_DownloadFileCompleted(object sender, AsyncCompletedEventArgs e)
        {
            TextResult.Text += " Download completed. \n";
        }
        public async Task GetMultipleWordCount()
        {
            var client = new WebClient();
            var results = new List<KeyValuePair<string, int>>();
            var urlList = GetBookUrls();
            var bookQuery = from book in urlList select ProcessBook2(book, cts.Token);
            var bookTasks = bookQuery.ToList();
            while (bookTasks.Count > 0)
            {
                var firstFinished = await Task.WhenAny(bookTasks);
                bookTasks.Remove(firstFinished);
                var thisBook = await firstFinished;
                TextResult.Text += String.Format("Finished downloading {0}. Word count: {1}\n",
                    thisBook.Key,
                    thisBook.Value);
            }
        }
        async Task<KeyValuePair<string, int>> ProcessBook(KeyValuePair<string, string> book, HttpClient client, CancellationToken ct)
        {
            await Task.Delay(500);
            try
            {
                TextResult.Text += String.Format("Getting the word count for {0}...\n", book.Key);
                HttpResponseMessage response = await client.GetAsync(book.Value, ct);
                var words = await response.Content.ReadAsStringAsync();
                var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                return new KeyValuePair<string, int>(book.Key, wordArray.Count());
            }
            catch (OperationCanceledException)
            {
                TextResult.Text += "Download cancelled.\n";
            }
            catch (Exception ex)
            {
                TextResult.Text += String.Format("An error has occurred: {0} \n", ex.Message);
            }
            return new KeyValuePair<string, int>("", 0);
        }
        private List<KeyValuePair<string, string>> GetBookUrls()
        {
            var baseUrl = "http://teknology360.com/data/Shakespeare/";

            var files = new List<string> { "allswellthatendswell",
                "amsnd", "antandcleo",
                "asyoulikeit", "comedyoferrors",
                "cymbeline", "hamlet",
                "henryiv1", "henryiv2",
                "henryv", "henryvi1",
                "henryvi2", "henryvi3",
                "henryviii", "juliuscaesar",
                "kingjohn", "kinglear",
                "loveslobourslost", "maan",
                "macbeth", "measureformeasure",
                "merchantofvenice", "othello",
                "richardii", "richardiii",
                "romeoandjuliet", "tamingoftheshrew",
                "tempest", "themwofw",
                "thetgofv", "timon",
                "titus", "troilusandcressida",
                "twelfthnight", "winterstale"
            };
            return files.Select(f => new KeyValuePair<string, string>(f, string.Format("{0}{1}.txt", baseUrl, f))).ToList();


            //var urlList = new List<KeyValuePair<string, string>>
            //{
            //    new KeyValuePair<string,string>("Origin of Species",
            //                "http://www.gutenberg.org/files/2009/2009.txt"),
            //    new KeyValuePair<string,string>("Beowulf",
            //                "http://www.gutenberg.org/files/16328/16328-8.txt"),
            //    new KeyValuePair<string,string>("Ulysses",
            //                "http://www.gutenberg.org/files/4300/4300.txt")
            //};
            //return urlList;
        }
        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            if (cts != null)
            {
                cts.Cancel();
            }
        }
    }
}