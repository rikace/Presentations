using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Reactive;
using System.Reactive.Linq;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Net;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;
using System.Windows.Forms.DataVisualization.Charting;
using System.Threading.Tasks.Dataflow;

namespace RxConcurrentStockTickers
{
    public class StockData
    {
        public StockData(string symbol, DateTime date, double open, double high, double low, double close)
        {
            Symbol = symbol;
            Date = date;
            Open = open;
            High = high;
            Low = low;
            Close = close;
        }

        public string Symbol { get; }
        public DateTime Date { get; set; }
        public Double Open { get; }
        public Double High { get; }
        public Double Low { get; }
        public Double Close { get; }

        public static StockData Parse(string symbol, string row)
        {
            if (string.IsNullOrWhiteSpace(row))
                return null;

            var cells = row.Split(',');
            if (!DateTime.TryParse(cells[0], out DateTime date))
                return null;

            var open = ParseDouble(cells[1]);
            var high = ParseDouble(cells[2]);
            var low = ParseDouble(cells[3]);
            var close = ParseDouble(cells[4]);
            return new StockData(symbol, date, open, high, low, close);
        }

        private static double ParseDouble(string s)
        {
            if (double.TryParse(s, out double x))
                return x;
            return -1;
        }
    }

    class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);

            var chart = new Chart { Dock = DockStyle.Fill };
            chart.ChartAreas.Add(new ChartArea("MainArea"));
            var form = new Form { Visible = true, Width = 1000, Height = 500 };
            form.Controls.Add(chart);

            var ctx = SynchronizationContext.Current;

            string[] stockFiles = new string[] { "aapl.csv", "amzn.csv", "fb.csv", "goog.csv", "msft.csv" };
            var sw = Stopwatch.StartNew();

            Task.Factory.StartNew(() =>
            {
                // Task
                // implement a stock analyzer using Reactive Extension
                // and the ObservableStreams extension method to parse the StockData
                // try different things such as, buffering or throttling 
                // Ultimately, subscribe to update the chart
                stockFiles
                    .ObservableStreams(StockData.Parse)
                    //  .Throttle(TimeSpan.FromMilliseconds(50))
                    .ObserveOn(ctx)
                    //   .Buffer(10)
                    .Subscribe(x =>
                    {
                        UpdateChart(chart, x, sw.ElapsedMilliseconds);
                        print(x);
                    });

                //stockFiles
                //    .ObservableStreams(StockData.Parse, 50)
                //    .GroupBy(stock =>
                //    {
                //   // .Throttle(TimeSpan.FromMilliseconds(100))
                // //       Thread.Sleep(100);
                //        return stock.Symbol;
                //    })
                //    .Subscribe(group =>
                //     {
                //         group                       
                //         .ObserveOn(ctx)
                //         // .Buffer(10)
                //         .Subscribe(x =>
                //         {
                //             UpdateChart(chart, x, sw.ElapsedMilliseconds);
                //             print(x, Thread.CurrentThread.ManagedThreadId);

                //         });
                //     });
            });

            Application.Run(form);
            Console.ReadLine();
        }



        static void print(IList<StockData> stocks) => printAction.Post(stocks);
        static void print(StockData stock) => printAction.Post(new List<StockData> { stock });


        private static ActionBlock<IList<StockData>> printAction = new ActionBlock<IList<StockData>>(data =>
        {
            var symbol = data.First().Symbol;
            ConsoleColor symbolColor = GetColorForSymbol(symbol.Substring(0, symbol.IndexOf('.')));
            using (new ColorPrint(symbolColor))
                foreach (var x in data)
                    Console.WriteLine($"{x.Symbol}({x.Date}) = {x.High}-{x.Low} {x.Open}/{x.Close}");
        });

        static ConsoleColor GetColorForSymbol(string symbol)
        {
            switch (symbol)
            {
                case "msft":
                    return ConsoleColor.Cyan;
                case "aapl":
                    return ConsoleColor.Red;
                case "fb":
                    return ConsoleColor.Magenta;
                case "goog":
                    return ConsoleColor.Yellow;
                case "amzn":
                    return ConsoleColor.Green;
                default:
                    return Console.ForegroundColor;
            }
        }

        private class ColorPrint : IDisposable
        {
            private ConsoleColor old;
            public ColorPrint(ConsoleColor color)
            {
                old = Console.ForegroundColor;
                Console.ForegroundColor = color;
            }
            public void Dispose()
            {
                Console.ForegroundColor = old;
            }
        }

        static void UpdateChart(Chart chart, IList<StockData> dd, long elapsedMilliseconds)
        {
            var symbol = dd.First().Symbol;
            Series series = chart.Series.FirstOrDefault(x => x.LegendText == symbol);
            if (series == null)
            {
                series = new Series
                {
                    LegendText = symbol,
                    ChartType = SeriesChartType.Candlestick
                };
                chart.Series.Add(series);

                chart.Legends.Add(new Legend(symbol));
            }
            foreach (var d in dd)
                series.Points.AddXY(d.Date, d.High, d.Low, d.Open, d.Close);

            chart.Titles.Clear();
            chart.Titles.Add($"Time elapsed {elapsedMilliseconds} ms");
        }

        static void UpdateChart(Chart chart, StockData d, long elapsedMilliseconds)
        {
            Series series = chart.Series.FirstOrDefault(x => x.LegendText == d.Symbol);
            if (series == null)
            {
                series = new Series
                {
                    LegendText = d.Symbol,
                    ChartType = SeriesChartType.Candlestick
                };
                chart.Series.Add(series);

                chart.Legends.Add(new Legend(d.Symbol));
            }

            series.Points.AddXY(d.Date, d.High, d.Low, d.Open, d.Close);

            chart.Titles.Clear();
            chart.Titles.Add($"Time elapsed {elapsedMilliseconds} ms");
        }
    }
}