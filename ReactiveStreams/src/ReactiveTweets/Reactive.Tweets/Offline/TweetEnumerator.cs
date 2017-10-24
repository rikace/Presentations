using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using Tweetinvi.Models;

namespace Reactive.Tweets
{
    public class TweetEnumerator : IEnumerator<ITweet>
    {
        private StreamReader _reader;
        private readonly bool loop;
        public TweetEnumerator(bool loop)
        {
            this.loop = loop;
            Reset();
        }

        public ITweet Current { get; private set; }

        object IEnumerator.Current => Current;

        public void Dispose()
        {
            _reader.Dispose();
        }

        public bool MoveNext()
        {
            var line = _reader.ReadLine();
            if (line != null)
            {
                var json = JObject.Parse(line);
                Current = new Tweet(json["TweetDTO"]);
            }
            if (loop && _reader.EndOfStream)
                _reader.BaseStream.Position = 0;
            return line != null;
        }

        public void Reset()
        {
            _reader?.Dispose();
            _reader = new StreamReader(@"../../tweets.txt");
        }
    }
}
