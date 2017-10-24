using System;
using System.Collections.Generic;
using System.Linq;
using Akka;
using Akka.Actor;
using Akka.Streams;
using Akka.Streams.Dsl;
using Tweetinvi.Models;

namespace Reactive.Tweets
{
    public static class TweetsToConsole
    {
        public static IRunnableGraph<TMat>
            CreateRunnableGraph<TMat>(Source<ITweet, TMat> tweetSource)
        {
            var formatFlow = Flow.Create<ITweet>().Select(Utils.FormatTweet);
            var writeSink = Sink.ForEach<string>(Console.WriteLine);
            return tweetSource.Via(formatFlow).To(writeSink);
        }
    }
}
