using System.Collections.Generic;
using GalaSoft.MvvmLight;
using OxyPlot;
using OxyPlot.Axes;
using ReactiveStock.ActorModel;
using ReactiveStock.ActorModel.Actors;
using ReactiveStock.ActorModel.Actors.UI;
using ReactiveStock.ActorModel.Actors.Core;

namespace ReactiveStock.ViewModel
{
    public class MainWindowViewModel : ViewModelBase
    {
        private IAgent<ChartSeriesMessage> _chartingActorRef;
        private IAgent<StocksCoordinatorMessage> _stocksCoordinatorActorRef;
        private PlotModel _plotModel;

        public Dictionary<string, StockToggleButtonViewModel> StockButtonViewModels { get; set; }

        public PlotModel PlotModel
        {
            get { return _plotModel; }
            set { Set(() => PlotModel, ref _plotModel, value); }
        }

        public MainWindowViewModel()
        {
            SetUpChartModel();

            InitializeActors();

            CreateStockButtonViewModels();
        }


        private void SetUpChartModel()
        {
            _plotModel = new PlotModel
            {
                LegendTitle = "Legend",
                LegendOrientation = LegendOrientation.Horizontal,
                LegendPlacement = LegendPlacement.Outside,
                LegendPosition = LegendPosition.TopRight,
                LegendBackground = OxyColor.FromAColor(200, OxyColors.White),
                LegendBorder = OxyColors.Black
            };


            var stockDateTimeAxis = new DateTimeAxis
            {
                Position = AxisPosition.Bottom,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                Title = "Date",
                StringFormat = "HH:mm:ss"
            };

            _plotModel.Axes.Add(stockDateTimeAxis);


            var stockPriceAxis = new LinearAxis
            {
                Minimum = 0,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                Title = "Price"
            };

            _plotModel.Axes.Add(stockPriceAxis);
        }


        private void InitializeActors()
        {
            _chartingActorRef =
                new LineChartingActor(PlotModel).Actor;

            _stocksCoordinatorActorRef =
                new StocksCoordinatorActor(_chartingActorRef).Actor;
        }


        private void CreateStockButtonViewModels()
        {
            StockButtonViewModels = new Dictionary<string, StockToggleButtonViewModel>();

            CreateStockButtonViewModel("AAAA");
            CreateStockButtonViewModel("BBBB");
            CreateStockButtonViewModel("CCCC");
        }

        private void CreateStockButtonViewModel(string stockSymbol)
        {
            var newViewModel = new StockToggleButtonViewModel(_stocksCoordinatorActorRef, stockSymbol);

            StockButtonViewModels.Add(stockSymbol, newViewModel);
        }
    }
}