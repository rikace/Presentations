using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive.Concurrency;
using System.Reactive.Linq;

namespace RxConcurrentStockTickers
{
    class FileLinesStream<T>
    {
        public FileLinesStream(string filePath, Func<string, T> map)
        {
            _filePath = filePath;
            _map = map;
            _data = new List<T>();
        }

        private string _filePath;
        private List<T> _data;
        private Func<string, T> _map;

        public IEnumerable<T> GetLines()
        {

            using (var stream = File.OpenRead(Path.Combine("Tickers", _filePath)))
            using (var reader = new StreamReader(stream))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var value = _map(line);
                    if (value != null)
                        _data.Add(value);
                }
            }
            _data.Reverse();
            while (true)
                foreach (var item in _data)
                    yield return item;
        }

        public IObservable<T> ObserveLines() => GetLines().ToObservable(TaskPoolScheduler.Default);
    }
    public static class ObservableDataStreams
    {
        public static IObservable<StockData> ObservableStreams
            (this IEnumerable<string> filePaths, Func<string, string, StockData> map, int delay = 50)
        {
            var flStreams =
                filePaths
                     .Select(x => new FileLinesStream<StockData>(x, row => map(x, row)))
                     .ToList();
            return
                flStreams
                    .Select(x =>
                    {
                        var startData = new DateTime(2001, 1, 1);
                        return Observable
                                .Interval(TimeSpan.FromMilliseconds(delay))
                                .Zip(x.ObserveLines(), (tick, stock) =>
                                {
                                    stock.Date = startData + TimeSpan.FromDays(tick);
                                    return stock;
                                });
                    }
                    )
                    // TASK merge all the Observable into one
                    .Aggregate((o1, o2) => o1.Merge(o2));
        }
    }
}
