using System;
using System.Collections.Generic;
using System.Linq;
using Akka;
using Akka.Streams;
using Akka.Streams.Dsl;
using Tweetinvi.Models;

namespace Reactive.Tweets
{
    public static class TweetsWithWeather
    {
        public static IRunnableGraph<TMat>
            CreateRunnableGraph<TMat>(Source<ITweet, TMat> tweetSource)
        {
            var formatUser = Flow.Create<IUser>()
                .Select(Utils.FormatUser);
            var formatTemperature = Flow.Create<decimal>()
                .Select(Utils.FormatTemperature);
            var writeSink = Sink.ForEach<string>(Console.WriteLine);

            var graph = GraphDsl.Create(b =>
            {
                var broadcast = b.Add(new Broadcast<ITweet>(2));
                var merge = b.Add(new Merge<string>(2));
                b.From(broadcast.Out(0))
                    .Via(Flow.Create<ITweet>().Select(tweet => tweet.CreatedBy)
                        .Throttle(10, TimeSpan.FromSeconds(1), 1, ThrottleMode.Shaping))
                    .Via(formatUser)
                    .To(merge.In(0));
                b.From(broadcast.Out(1))
                    .Via(Flow.Create<ITweet>().Select(tweet => tweet.Coordinates)
                        .Buffer(10, OverflowStrategy.DropNew)
                        .Throttle(1, TimeSpan.FromSeconds(1), 10, ThrottleMode.Shaping))
                    .Via(Flow.Create<ICoordinates>().SelectAsync(5, Utils.GetWeatherAsync))
                    .Via(formatTemperature)
                    .To(merge.In(1));

                return new FlowShape<ITweet, string>(broadcast.In, merge.Out);
            });

            return tweetSource.Where(x=>x.Coordinates != null)
                              .Via(graph).To(writeSink);
        }
    }
}
