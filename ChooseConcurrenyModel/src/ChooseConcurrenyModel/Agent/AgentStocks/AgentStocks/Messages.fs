module Messages

open System

type StocksCoordinatorMessage =
    | WatchStock of string
    | UnWatchStock of string

type ChartSeriesMessage =
    | AddSeriesToChart of string //AddSeriesToChart
    | RemoveSeriesFromChart of string //RemoveSeriesFromChart
    | HandleStockPrice of string * decimal * DateTime // HandleNewStockPrice

type StockAgentMessage = 
    | SubscribeStockPrices of string * MailboxProcessor<ChartSeriesMessage> 
    | UnSubscribeStockPrices of string
    | UpdateStockPrices of decimal * DateTime

type StockPriceLookupMessage =
    | RefreshStockPrice of AsyncReplyChannel<decimal * DateTime>

type FlipToggleMessage =
    | FlipToggle 
