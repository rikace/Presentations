using System;
using System.Collections.Generic;
using System.Linq;
using Akka;
using Akka.Streams;
using Akka.Streams.Dsl;
using Tweetinvi.Models;
using Shared.Reactive;


namespace Reactive.Tweets
{
    public static class TweetsWithBroadcast
    {
        public static IRunnableGraph<TMat>
            CreateRunnableGraph<TMat>(Source<ITweet, TMat> tweetSource)
        {
            var formatUser = Flow.Create<IUser>().Select(Utils.FormatUser);
            var formatCoordinates = Flow.Create<ICoordinates>().Select(Utils.FormatCoordinates);

            var writeSink = Sink.ForEach<string>(Console.WriteLine);

            var graph = GraphDsl.Create(buildBlock: b =>
            {
                var broadcast = b.Add(new Broadcast<ITweet>(2));
                var merge = b.Add(new Merge<string>(2));
                b.From(broadcast.Out(0))
                    .Via(Flow.Create<ITweet>().Select(tweet => tweet.CreatedBy))
                    .Via(formatUser)
                    .To(merge.In(0));

                b.From(broadcast.Out(1))
                    .Via(Flow.Create<ITweet>().Select(tweet => tweet.Coordinates))
                    .Via(formatCoordinates)
                    .To(merge.In(1));

                return new FlowShape<ITweet, string>(broadcast.In, merge.Out);
            });

            return tweetSource.Where(x => x.Coordinates != null)
                              .Via(graph).To(writeSink);
        }
    }
}
