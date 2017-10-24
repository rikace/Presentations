using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using ReactiveStock.ActorModel.Messages;
using ReactiveStock.ActorModel.Actors.Core;

namespace ReactiveStock.ActorModel.Actors.UI
{
    class LineChartingActor
    {
        private readonly PlotModel _chartModel;
        private readonly Dictionary<string, LineSeries> _series;

        public IAgent<ChartSeriesMessage> Actor { get; private set; }

        public LineChartingActor(PlotModel chartModel)
        {
            _chartModel = chartModel;
            _series = new Dictionary<string, LineSeries>();

            Actor = Agent.Start<ChartSeriesMessage>(message =>
                {
                    switch (message)
                    {
                        case AddChartSeriesMessage msg:
                            AddSeriesToChart(msg);
                            break;
                        case RemoveChartSeriesMessage msg:
                            RemoveSeriesFromChart(msg);
                            break;
                        case StockPriceMessage msg:
                            HandleNewStockPrice(msg);
                            break;
                        default:
                            throw new ArgumentException(
                               message: "message is not a recognized",
                               paramName: nameof(message));
                    }
                });
        }

        private void AddSeriesToChart(AddChartSeriesMessage message)
        {
            if (!_series.ContainsKey(message.StockSymbol))
            {
                var newLineSeries = new LineSeries
                {
                    StrokeThickness = 2,
                    MarkerSize = 3,
                    MarkerStroke = OxyColors.Black,
                    MarkerType = MarkerType.None,
                    CanTrackerInterpolatePoints = false,
                    Title = message.StockSymbol,
                    Smooth = false
                };


                _series.Add(message.StockSymbol, newLineSeries);

                _chartModel.Series.Add(newLineSeries);

                RefreshChart();
            }
        }

        private void RemoveSeriesFromChart(RemoveChartSeriesMessage message)
        {
            if (_series.ContainsKey(message.StockSymbol))
            {
                var seriesToRemove = _series[message.StockSymbol];

                _chartModel.Series.Remove(seriesToRemove);

                _series.Remove(message.StockSymbol);

                RefreshChart();
            }
        }

        private void HandleNewStockPrice(StockPriceMessage message)
        {
            if (_series.ContainsKey(message.StockSymbol))
            {
                var series = _series[message.StockSymbol];

                var newDataPoint = new DataPoint(DateTimeAxis.ToDouble(message.Date),
                    LinearAxis.ToDouble(message.StockPrice));

                // Keep the last 10 data points on graph
                if (series.Points.Count > 10)
                {
                    series.Points.RemoveAt(0);
                }

                series.Points.Add(newDataPoint);

                RefreshChart();
            }
        }


        private void RefreshChart()
        {
            _chartModel.InvalidatePlot(true);
        }
    }
}
