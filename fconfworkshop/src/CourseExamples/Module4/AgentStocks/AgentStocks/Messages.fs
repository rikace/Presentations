module Messages

open System

type StocksCoordinatorMessage =
    | WatchStock of string
    | UnWatchStock of string

type ChartSeriesMessage =
    | AddSeriesToChart of string  
    | RemoveSeriesFromChart of string 
    | HandleStockPrice of string * decimal * DateTime 

type StockAgentMessage = { Price:decimal; Time:DateTime }

type StockPriceLookupMessage =
    | RefreshStockPrice of AsyncReplyChannel<decimal * DateTime>

type FlipToggleMessage =
    | FlipToggle 
