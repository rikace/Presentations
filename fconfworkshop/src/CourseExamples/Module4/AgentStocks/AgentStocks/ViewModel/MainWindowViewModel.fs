namespace FSharpWpfMvvmTemplate.ViewModel

open System
open System.Collections.ObjectModel
open System.Windows
open System.Windows.Data
open System.Windows.Input
open System.ComponentModel
open System.Collections.Generic
open OxyPlot
open OxyPlot.Axes
open OxyPlot.Series
open Messages


type MainWindowViewModel() =
    inherit ViewModelBase()

    let mutable plotModel =
        let p = PlotModel()
        p.LegendTitle <- "Legend"
        p.LegendOrientation <- LegendOrientation.Horizontal
        p.LegendPlacement <- LegendPlacement.Outside
        p.LegendPosition <- LegendPosition.TopRight
        p.LegendBackground <- OxyColor.FromAColor(200uy, OxyColors.White)
        p.LegendBorder <- OxyColors.Black
        p

    let mutable stockButtonViewModels = Dictionary<string, StockToggleButtonViewModel>()

    let chartingActorRef = Agents.lineChartingAgent(plotModel)
    let stocksCoordinatorActorRef = Agents.stocksCoordinatorAgent(chartingActorRef)
    
    do
        let stockDateTimeAxis = new DateTimeAxis()
        stockDateTimeAxis.Position <- AxisPosition.Bottom
        stockDateTimeAxis.MajorGridlineStyle <- LineStyle.Solid
        stockDateTimeAxis.MinorGridlineStyle <- LineStyle.Dot
        stockDateTimeAxis.Title <- "Date"
        stockDateTimeAxis.StringFormat <- "HH:mm:ss"
        plotModel.Axes.Add(stockDateTimeAxis)

        let stockPriceAxis = new LinearAxis()
        stockPriceAxis.Minimum <- 0.
        stockPriceAxis.MajorGridlineStyle <- LineStyle.Solid
        stockPriceAxis.MinorGridlineStyle <- LineStyle.Dot
        stockPriceAxis.Title <- "Price"
        plotModel.Axes.Add(stockPriceAxis)

        for stock in ["MSFT"; "APPL"; "GOOG"] do
            let  s = StockToggleButtonViewModel(stock, stocksCoordinatorActorRef)
            stockButtonViewModels.Add(stock, s)
        
    member self.StockButtonViewModels 
        with get() = stockButtonViewModels
        and set value = stockButtonViewModels <- value

    member self.PlotModel
        with get() = plotModel
        and set value = 
            plotModel <- value
            base.OnPropertyChanged("PlotModel")

        
