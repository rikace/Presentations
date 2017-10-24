using System;
using System.Collections.Generic;
using System.Configuration;
using Akka.Actor;
using Akka.Streams;
using Tweetinvi;
using Tweetinvi.Models;
using Akka.Streams.Dsl;

namespace Reactive.Tweets
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            using (var sys = ActorSystem.Create("Reactive-Tweets"))
            {
                var consumerKey = ConfigurationManager.AppSettings["ConsumerKey"];
                var consumerSecret = ConfigurationManager.AppSettings["ConsumerSecret"];
                var accessToken = ConfigurationManager.AppSettings["AccessToken"];
                var accessTokenSecret = ConfigurationManager.AppSettings["AccessTokenSecret"];

                Console.OutputEncoding = System.Text.Encoding.UTF8;
                Console.ForegroundColor = ConsoleColor.Cyan;

                Console.WriteLine("Press Enter to Start");
                Console.ReadLine();

                var useCachedTweets = true;
                
                using (var mat = sys.Materializer())
                {
                    Auth.SetCredentials(new TwitterCredentials(consumerKey, consumerSecret, accessToken, accessTokenSecret));

                    if (useCachedTweets)
                    {
                        var tweetSource = Source.FromEnumerator(() => new TweetEnumerator(true));
                        var graph = CreateRunnableGraph(tweetSource);
                        graph.Run(mat);
                    }
                    else
                    {
                        var tweetSource = Source.ActorRef<ITweet>(100, OverflowStrategy.Backpressure);
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

                =>   //TweetsToConsole.CreateRunnableGraph(tweetSource);
                     //TweetsWithBroadcast.CreateRunnableGraph(tweetSource);
                    // TweetsWithThrottle.CreateRunnableWethaerGraph(tweetSource);
                     //TweetsWithThrottle.CreateRunnableGraph(tweetSource);
                  //   TweetsWithWeather.CreateRunnableGraph(tweetSource);
                     TweetsToEmotion.CreateRunnableGraph(tweetSource);

        }
    }
}
