using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Xml.Linq;
using Akka.Actor;
using Akka.Streams;
using Tweetinvi;
using Tweetinvi.Models;
using Akka.Streams.Dsl;
using Newtonsoft.Json;
using Shared.Reactive;
using Shared.Reactive.Tweets;

namespace Reactive.Tweets
{
    class TweetCoordinates
    {

        public double Longitude { get; set; }
        public double Latitude { get; set; }
        public decimal Temp { get; set; }
    }

    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            using (var sys = ActorSystem.Create("Reactive -Tweets"))
            {
                var consumerKey = ConfigurationManager.AppSettings["ConsumerKey"];
                var consumerSecret = ConfigurationManager.AppSettings["ConsumerSecret"];
                var accessToken = ConfigurationManager.AppSettings["AccessToken"];
                var accessTokenSecret = ConfigurationManager.AppSettings["AccessTokenSecret"];

                Console.OutputEncoding = System.Text.Encoding.UTF8;
                Console.ForegroundColor = ConsoleColor.Cyan;

                Console.WriteLine("Press Enter to Start");
                Console.ReadLine();

                var useCachedTweets = false;

                using (var mat = sys.Materializer())
                {

                    if (useCachedTweets)
                    {
                        var tweetSource = Source.FromEnumerator(() => new TweetEnumerator(true));
                        var graph = CreateRunnableGraph(tweetSource);
                        graph.Run(mat);
                    }
                    else
                    {
                        Auth.SetCredentials(new TwitterCredentials(consumerKey, consumerSecret, accessToken, accessTokenSecret));

                        var tweetSource = Source.ActorRef<ITweet>(100, OverflowStrategy.DropBuffer);
                        var graph = CreateRunnableGraph(tweetSource);
                        var actor = graph.Run(mat);
                        Utils.StartSampleTweetStream(actor);
                    }

                    Console.WriteLine("Press Enter to exit");
                    TweetsToEmotion.form?.ShowDialog();
                    Console.ReadLine();
                }
            }

            IRunnableGraph<TMat> CreateRunnableGraph<TMat>(Source<ITweet, TMat> tweetSource)

                =>   //  TweetsToConsole.CreateRunnableGraph(tweetSource);
                     // TweetsWithBroadcast.CreateRunnableGraph(tweetSource);
                  //    TweetsWithThrottle.CreateRunnableGraph(tweetSource);
                      TweetsWithThrottle.CreateRunnableWeatherGraph(tweetSource);
                  //    TweetsWithWeather.CreateRunnableGraph(tweetSource);
                      //TweetsToEmotion.CreateRunnableGraph(tweetSource);

        }
    }
}
