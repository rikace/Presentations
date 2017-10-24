using Akka;
using Akka.Streams.Dsl;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tweetinvi.Models;
using LiveCharts;
using LiveCharts.WinForms;
using System.Windows.Forms;
using System.Windows.Threading;

namespace Reactive.Tweets
{
    public static class TweetsToEmotion
    {
        public static IRunnableGraph<TMat>
            CreateRunnableGraph<TMat>(Source<ITweet, TMat> tweetSource)
        {
            var emotionAnalysis = new EmotionAnalysis("../../models");
            var formatFlow = Flow.Create<ITweet>().Select(emotionAnalysis.AddEmotion);

            var chart = DoughnutChart();
            InitForm(chart);

            var writeSink = Sink.ForEach<(ITweet, EmotionAnalysis.Emotion)>(
                (tuple) =>
                {
                    var (tweet, emotion) = tuple;
                    UpdateChart(chart, emotion.ToString());
                    Console.WriteLine($"[{emotion.ToString()}] - {tweet.Text}");
                });
            return tweetSource.Via(formatFlow).To(writeSink);
        }


        public static Form form;
        private static Dispatcher dispatcher;

        private static PieChart DoughnutChart()
        {
            return new PieChart
            {
                InnerRadius = 100,
                LegendLocation = LegendLocation.Right,
                Series = new SeriesCollection
                {
                    CreatePieSeries("Unhappy"),
                    CreatePieSeries("Indifferent"),
                    CreatePieSeries("Happy")
                }
            };

            LiveCharts.Wpf.PieSeries CreatePieSeries(string title)
            {
                return new LiveCharts.Wpf.PieSeries
                {
                    Title = title,
                    Values = new ChartValues<int> { 0 },
                    DataLabels = true
                };
            }
        }

        private static void InitForm(Control control)
        {
            form = new Form() { Text = "Emotions", Width = 400, Height = 300 };
            control.Dock = DockStyle.Fill;
            form.Controls.Add(control);

            dispatcher = Dispatcher.CurrentDispatcher;
        }

        private static void UpdateChart(PieChart chart, string emotion)
        {
            dispatcher.Invoke(() =>
            {
                var series = chart.Series.First(x => x.Title == emotion).Values;
                series[0] = 1 + ((int)series[0]);
            });
        }
    }
}
