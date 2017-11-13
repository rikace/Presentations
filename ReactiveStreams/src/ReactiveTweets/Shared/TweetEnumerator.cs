using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using Tweetinvi.Models;

namespace Shared.Reactive.Tweets
{
    public class TweetEnumerator : IEnumerator<ITweet>
    {
        private StreamReader _reader;
        private readonly bool loop;
        private readonly string _filePath;
        public TweetEnumerator(bool loop, string filePath=@"../../../Tweets/tweets_cu.txt")
        {
            this.loop = loop;
            this._filePath = filePath;
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
            _reader = new StreamReader(_filePath);
        }
    }
}
