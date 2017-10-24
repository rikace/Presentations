using System;
using System.Collections.Generic;
using System.Linq;
using Akka;
using Akka.Streams;
using Akka.Streams.Dsl;
using Tweetinvi.Models;

namespace Reactive.Tweets
{
    public static class TweetsWithThrottle
    {
        public static IRunnableGraph<TMat>
            CreateRunnableGraph<TMat>(Source<ITweet, TMat> tweetSource)
        {
            var formatUser = Flow.Create<IUser>()
                .Select(Utils.FormatUser);
            var formatCoordinates = Flow.Create<ICoordinates>()
                .Select(Utils.FormatCoordinates);
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
                        //.Buffer(10, OverflowStrategy.DropNew)
                        .Throttle(1, TimeSpan.FromSeconds(1), 10, ThrottleMode.Shaping))
                    .Via(formatCoordinates)
                    .To(merge.In(1));

                return new FlowShape<ITweet, string>(broadcast.In, merge.Out);
            });

            return tweetSource.Where(x => x.Coordinates != null)
                              .Via(graph).To(writeSink);
        }

        public static IRunnableGraph<TMat>
            CreateRunnableWethaerGraph<TMat>(Source<ITweet, TMat> tweetSource)
        {
            var formatUser = Flow.Create<IUser>()
                .Select(Utils.FormatUser);
            var formatCoordinates = Flow.Create<ICoordinates>()
                .Select(Utils.FormatCoordinates);
            var formatTemperature = Flow.Create<decimal>()
              .Select(Utils.FormatTemperature);
            var writeSink = Sink.ForEach<string>(Console.WriteLine);

            // 1- Throttle at the same rate line 72 (Throttle(10))
            // 2- Throttle at different rate line 72 (Throttle(1))
            //     only 1 message because we have 1 stream source & broadcast = 2 channel with 1 request with 10 msg per second and 1 request with 1 msg per second... but we have only 1 stream source, so it cannot send messages to a different rate thus it sattisfy the lowest requirement.
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
                        //.Buffer(10, OverflowStrategy.DropNew)
                        .Throttle(1, TimeSpan.FromSeconds(1), 1, ThrottleMode.Shaping))
                        .Via(Flow.Create<ICoordinates>().SelectAsync(5, Utils.GetWeatherAsync))
                    .Via(formatTemperature)
                    .To(merge.In(1));

                return new FlowShape<ITweet, string>(broadcast.In, merge.Out);
            });

            return tweetSource.Where(x => x.Coordinates != null)
                              .Via(graph).To(writeSink);
        }
    }
}
