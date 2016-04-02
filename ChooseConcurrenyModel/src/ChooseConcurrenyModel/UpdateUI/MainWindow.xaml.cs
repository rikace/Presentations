using System;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using System.Windows;

namespace UpdateUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }
        private void Button_Click_1(object sender, RoutedEventArgs e)
        {
            Task.Factory.StartNew(() =>
            {
                char[] delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
                var client = new WebClient();
                const string headerText = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; Trident/6.0)";
                client.Headers.Add("user-agent", headerText);
                try
                {
                    var words = client.DownloadString(@"http://www.gutenberg.org/files/2009/2009.txt");
                    var wordArray = words.Split(delimiters, StringSplitOptions.RemoveEmptyEntries);
                    return wordArray;
                }
                finally
                {
                    client.Dispose();
                }
            }).ContinueWith(antecedent =>
            {
                lblWordCount.Content = String.Concat("Origin of Species word count: ",
                                                     antecedent.Result.Count().ToString());
            }, TaskScheduler.FromCurrentSynchronizationContext());
        }
    }
}
