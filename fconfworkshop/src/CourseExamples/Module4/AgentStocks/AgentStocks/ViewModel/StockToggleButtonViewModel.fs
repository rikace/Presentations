namespace FSharpWpfMvvmTemplate.ViewModel

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
open Messages
open Utils


type StockToggleButtonViewModel(stockSymbol:string, stockCoordinator:MailboxProcessor<StocksCoordinatorMessage>) as self =
    inherit ViewModelBase()

    let toggleButtonText isToggledOn =
                sprintf "%s %s" stockSymbol (if isToggledOn = true then "(on)" else "(off)")

    let mutable buttonText = toggleButtonText true
    let mutable stockSymbol = stockSymbol

    let toggleAgent =
            Agent.Start(fun inbox -> 
                let rec toggledOn() = async {
                    let! msg = inbox.Receive()
                    stockCoordinator.Post(UnWatchStock(stockSymbol))
                    self.UpdateButtonTextToOn()
                    return! toggledOff() }
                and toggledOff() = async {
                    let! msg = inbox.Receive()
                    stockCoordinator.Post(WatchStock(stockSymbol))
                    self.UpdateButtonTextToOff()
                    return! toggledOn()
                }
                toggledOff() )

    member x.ButtonText 
        with get() = buttonText
        and set value = 
            buttonText <- value
            base.OnPropertyChanged("ButtonText")
    
    member x.StockSymbol 
        with get() = stockSymbol
        and set value =
            stockSymbol <- value

    member x.UpdateButtonTextToOff() =        
            x.ButtonText <- toggleButtonText(false)

    member x.UpdateButtonTextToOn() =        
            x.ButtonText <- toggleButtonText(true)
            
    member x.StockToggleButtonActorRef
        with get() =  toggleAgent       


    member x.ToggleCommand = 
        new RelayCommand ((fun canExecute -> true), 
            (fun _ -> x.StockToggleButtonActorRef.Post(FlipToggle))) 



