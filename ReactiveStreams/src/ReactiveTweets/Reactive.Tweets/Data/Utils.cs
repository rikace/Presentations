using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using System.Threading.Tasks;
using Akka.Actor;
using Newtonsoft.Json;
using Tweetinvi;
using Tweetinvi.Models;
using Tweetinvi.Models.Entities;
using Stream = Tweetinvi.Stream;
using System.Net.Http;

namespace Reactive.Tweets
{
    public static class Utils
    {
        public static void StartSampleTweetStream(IActorRef actor, LanguageFilter langFilter = LanguageFilter.English)
        {
            var stream = Stream.CreateSampleStream();
            stream.AddTweetLanguageFilter(langFilter);

            stream.TweetReceived += (_, arg) =>
            {
                if (arg.Tweet.Coordinates != null)
                {
                    arg.Tweet.Text = arg.Tweet.Text.Replace("\r", " ").Replace("\n", " ");
                    SaveTweet(arg.Tweet);

                    actor.Tell(arg.Tweet);
                }
            };
            stream.StartStream();
        }

        public static void StartFilteredTweetStream(IActorRef actor, string track, LanguageFilter langFilter = LanguageFilter.English)
        {
            var stream = Stream.CreateFilteredStream();
            //var centerOfNewYork = new Location(new Coordinates(-75,38), new Coordinates(-71,43));
            //stream.AddLocation(centerOfNewYork);

            stream.AddTrack(track);
            stream.AddTweetLanguageFilter(langFilter);

            stream.MatchingTweetReceived += (_, arg) =>
            {
                arg.Tweet.Text = arg.Tweet.Text.Replace("\r", " ").Replace("\n", " ");
                SaveTweet(arg.Tweet);
                actor.Tell(arg.Tweet);
            };
            stream.StartStreamMatchingAnyCondition();
        }

        private static void SaveTweet(ITweet tweet)
        {
            if (tweet.Coordinates == null)
                return;

            var json = JsonConvert.SerializeObject(tweet);
            File.AppendAllText("../../tweets.txt", $"{json}\r\n");
        }

        public static async Task<decimal> GetWeatherAsync(ICoordinates coordinates)
        {
            using (var httpClient = new HttpClient())
            {
                var requestUrl = $"http://api.met.no/weatherapi/locationforecast/1.9/?lat={coordinates.Latitude};lon={coordinates.Latitude}";
                var result = await httpClient.GetStringAsync(requestUrl);
                var doc = XDocument.Parse(result);
                var temp = doc.Root.Descendants("temperature").First().Attribute("value").Value;
                return decimal.Parse(temp);
            }
        }

        public static string FormatTweet(ITweet tweet)
        {
            var builder = new StringBuilder();
            builder.AppendLine("---------------------------------------------------------");
            builder.AppendLine($"Tweet from {tweet.CreatedBy} at {tweet.Coordinates?.Latitude},{tweet.Coordinates?.Longitude}");
            builder.AppendLine(tweet.Text);
            return builder.ToString();
        }

        public static string FormatUser(IUser user)
        {
            return user.ToString();
        }

        public static string FormatCoordinates(ICoordinates coordinates)
        {
            return $"------------------------------------{coordinates?.Latitude},{coordinates?.Longitude}";
        }

        public static string FormatTemperature(decimal temperature)
        {
            return $"------------------------------------{temperature}° Celcius";
        }
    }
}
