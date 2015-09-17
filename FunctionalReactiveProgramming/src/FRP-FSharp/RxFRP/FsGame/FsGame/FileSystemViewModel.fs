namespace ViewModels

open System
open System.Windows
open FSharp.ViewModule
open FSharp.ViewModule.Validation
open FsXaml
open System.Reactive
open System.Reactive.Concurrency
open System.Reactive.Linq
open System.Collections.Generic
open System.Collections.ObjectModel
open System.IO
open System.Threading

type FileSystemView = XAML<"FileSystemDemo.xaml", true>

type FileSystemModule() = 
    
    static member WhatsHappeningIn(path) =
        let watcher = new FileSystemWatcher()
        watcher.Path <- path

        let changes = 
              List.reduce Observable.merge 
                [ watcher.Changed
                  watcher.Created
                  watcher.Deleted               
                  (watcher.Renamed :?> IObservable<FileSystemEventArgs>) ]

        watcher.EnableRaisingEvents <- true

        changes

type ChangeDetails = { ChangeType:WatcherChangeTypes; Name:string }

type FileSystemViewModel() as self = 
    inherit ViewModelBase()   
    
    let mutable lastChange = ""
    let events = new ObservableCollection<string>()


    do
        let path = @"c:\temp"
        let folderWatching = FileSystemModule.WhatsHappeningIn(path)

        let myFile =
            folderWatching
            |> Observable.filter(fun f -> f.ChangeType = WatcherChangeTypes.Created)
            |> Observable.map(fun f -> {ChangeType=f.ChangeType; Name=f.Name})
            |> Observable.filter(fun f -> f.Name.StartsWith("my"))
        
        folderWatching
            |> Observable.map(fun f -> sprintf "%A : %s" f.ChangeType f.Name)
            |> Observable.subscribe(fun f-> self.LastChange <- f)
            |> ignore


        folderWatching
            |> Observable.map(fun f -> sprintf "%A : %s" f.ChangeType f.Name)
            |> fun f -> Observable.Timestamp(f)
            |> fun f-> Observable.DistinctUntilChanged(f, (fun e -> sprintf "%s%s" e.Value, e.Timestamp.ToString("YYYYMMDDhhmmsssf")))
            |> fun f -> Observable.Select(f, fun f -> f.Value)
            |> fun f -> DispatcherObservable.ObserveOnDispatcher(f)
            |> Observable.subscribe(fun f-> self.AddEvent f)
            |> ignore

    member self.Events
        with get() = events
    
    member self.AddEvent s =
        self.Events.Add s

    member self.LastChange 
        with get() = lastChange
        and set value = 
            lastChange <- value
            self.RaisePropertyChanged("LastChange")


