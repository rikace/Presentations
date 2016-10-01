module Agents 

    open System
    open System.Collections.Generic
    open System.Collections.ObjectModel
    open System.Windows
    open System.Windows.Data
    open System.Windows.Input
    open System.ComponentModel
    open OxyPlot
    open OxyPlot.Axes
    open OxyPlot.Series
    open System.Reactive.Linq
    open Messages


    type Agent<'a> = MailboxProcessor<'a>

    type private ThreadSafeRandomRequest =
    | GetDouble of AsyncReplyChannel<decimal>
    let private threadSafeRandomAgent = Agent.Start(fun inbox -> 
            let rnd = new Random()
            let rec loop() = async {
                let! GetDouble(reply) = inbox.Receive() 
                reply.Reply((rnd.Next(-5, 5) |> decimal))
                return! loop()
            }
            loop() )


    let updatePrice (price:decimal) =
                let newPrice' = price + (threadSafeRandomAgent.PostAndReply(GetDouble))
                if newPrice' < 0m then 5m
                elif newPrice' > 50m then 45m
                else newPrice'


    let stocksObservable (agent: MailboxProcessor<StockAgentMessage>) =
        Observable.Interval(TimeSpan.FromMilliseconds 150.)
        |> Observable.scan(fun s i -> updatePrice s) 20m
        |> Observable.add(fun u -> agent.Post(UpdateStockPrices(u, DateTime.Now)))

    let stockAgent (stockSymbol:string) =
        MailboxProcessor<StockAgentMessage>.Start(fun inbox -> 
            let subscribers =  Dictionary<string, MailboxProcessor<ChartSeriesMessage>>() 
            let rec loop stockPrice = async {
                let! msg = inbox.Receive()
                match msg with
                | UpdateStockPrices(p, d) ->                    
                    for subscriber in subscribers do
                        let message = HandleStockPrice(subscriber.Key, p, d)
                        subscriber.Value.Post(message)
                    return! loop p
                | SubscribeStockPrices(s, a) -> 
                    if not <| subscribers.ContainsKey(s) then                        
                        subscribers.Add(s,  a)
                    return! loop stockPrice
                | UnSubscribeStockPrices(s) -> 
                    if subscribers.ContainsKey(s) then                        
                        let agentToRemove = subscribers.[s]
                        (agentToRemove :> IDisposable).Dispose()
                        subscribers.Remove(s) |> ignore
                    return! loop stockPrice }
            loop 20m)

    let stocksCoordinatorAgent(lineChartingAgent:MailboxProcessor<ChartSeriesMessage>) =
        let stockAgents = Dictionary<string, MailboxProcessor<StockAgentMessage>>()
        MailboxProcessor<StocksCoordinatorMessage>.Start(fun inbox ->
                let rec loop() = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | WatchStock(s) -> 
                        if not <| stockAgents.ContainsKey(s) then 
                            let stockAgentChild = stockAgent(s)
                            stockAgents.Add(s, stockAgentChild)
                            stockAgents.[s].Post(SubscribeStockPrices(s, lineChartingAgent))
                            lineChartingAgent.Post(AddSeriesToChart(s))          
                            stocksObservable(stockAgentChild)
                        return! loop()                                         
                    | UnWatchStock(s) ->
                        if stockAgents.ContainsKey(s) then
                             lineChartingAgent.Post(RemoveSeriesFromChart(s))  
                             stockAgents.[s].Post(UnSubscribeStockPrices(s))
                             (stockAgents.[s] :> IDisposable).Dispose()
                             stockAgents.Remove(s) |> ignore
                        return! loop() } 
                loop() )


    let lineChartingAgent(chartModel:PlotModel) =
        let refreshChart() = chartModel.InvalidatePlot(true)
        MailboxProcessor<ChartSeriesMessage>.Start(fun inbox ->
                let series =  Dictionary<string, LineSeries>()
                let rec loop() = async {
                    let! msg = inbox.Receive()
                    match msg with
                    | AddSeriesToChart(s) -> 
                            if not <| series.ContainsKey(s) then
                                let lineSeries = LineSeries()
                                lineSeries.StrokeThickness <- 2.
                                lineSeries.MarkerSize <- 3.
                                lineSeries.MarkerStroke <- OxyColors.Black
                                lineSeries.MarkerType <- MarkerType.None
                                lineSeries.CanTrackerInterpolatePoints <- false
                                lineSeries.Title <- s
                                lineSeries.Smooth <- false
                                series.Add (s, lineSeries)
                                chartModel.Series.Add lineSeries
                                refreshChart()
                                return! loop()
                    | RemoveSeriesFromChart(s) -> 
                            if series.ContainsKey(s) then
                                let seriesToRemove = series.[s]
                                chartModel.Series.Remove(seriesToRemove) |> ignore
                                series.Remove(s) |> ignore
                                refreshChart()
                            return! loop()
                    | HandleStockPrice(s,p, d) -> 
                            if series.ContainsKey(s) then
                                let newDataPoint = new DataPoint(DateTimeAxis.ToDouble(d), LinearAxis.ToDouble(p))
                                let seriesToUpdate = series.[s]
                                if seriesToUpdate.Points.Count > 10 then
                                    seriesToUpdate.Points.RemoveAt(0)
                                seriesToUpdate.Points.Add(newDataPoint)
                                series.[s] <- seriesToUpdate
                                refreshChart()                            
                            return! loop() }
                loop() )