using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DataFLowTests
{

//    return quotes
//    .GroupBy(q => q.Symbol)
//    .SelectMany(g => g.Buffer(2, 1).Select(b => new PriceChange {
//        Symbol = b[0].Symbol,
//        Change = (b[1].Price - b[0].Price) / b[0].Price
//     })
//    );


//    public class StockTicker
//    {
//        public event EventHandler<PriceChangedEventArgs> OnPriceChanged;
//    }

//    public class TestSTockTicker
//    {
//        StockTicker stockTicker = 
//  new StockTicker("MSFT", "APPL", "YHOO", "GOOG");
//stockTicker.OnPriceChanged += (sender, args ) => 
//{
//  Console.WriteLine(
//    "{0}: Price - {1}  Volume - {2}", 
//    args.StockSymbol, args.Price, args.Volume);
//};

//    ////


//   StockTicker stockTicker = 
//  new StockTicker("MSFT", "APPL", "YHOO", "GOOG");

//IObservable<PriceChangedEventArgs> priceChangedObservable = 
//  Observable.FromEventPattern<PriceChangedEventArgs>(
//    eventHandler => stockTicker.OnPriceChanged += eventHandler,
//    eventHandler => stockTicker.OnPriceChanged -= eventHandler )
//      .Select( eventPattern => eventPattern.EventArgs );
//priceChangedObservable.Subscribe(args => Console.WriteLine( 
//  "{0}: Price - {1}  Volume - {2}", 
//    args.StockSymbol, args.Price, args.Volume ) );


//    priceChangedObservable
//  .Where(stock => stock.Volume >= minimumVolumeBoundary )
//  .Subscribe(args => Console.WriteLine( 
//    "{0}: Price - {1}  Volume - {2}", 
//    args.StockSymbol, args.Price, args.Volume ) );
//priceChangedObservable
//  .Select(stock => new {stock.StockSymbol, stock.Price} )
//  .Subscribe( args => Console.WriteLine( "{0}: Price - {1}",
//    args.StockSymbol, args.Price ) );




//priceChangedObservable
//  .Where(stock => stock.Volume >= minimumVolumeBoundary )
//  .Throttle( TimeSpan.FromSeconds(1) )
//  .Select(stock => new {stock.StockSymbol, stock.Price}  )
//  .Subscribe( args => Console.WriteLine( /* output */ ) );





//priceChangedObservable
//  .Select( stock => new { stock.StockSymbol, stock.Price } )
//  .GroupBy( stock => stock.StockSymbol )
//  .Subscribe( group =>
//    group.Throttle( TimeSpan.FromSeconds( 1 ) )
//      .Subscribe( stock => Console.WriteLine( /* ... */ ) ) );

//    }

//private ObservableCollection<StockChange> _StockChanges = 
//  new ObservableCollection<StockChange>();

//public ObservableCollection<StockChange> StockChanges
//{
//  get { return _StockChanges; }
//  set { _StockChanges = value; }
//} 

//StockChanges is an ObservableCollection of StockChange:
//public class StockChange
//{
//  public string StockSymbol { get; set; }
//  public double Price { get; set; }
//  public string Display
//  {
//    get { return string.Format( "{0,-10}   {1,-20:C}",
//      StockSymbol, Price); }
//  }
//} 



//public class StockViewModel
//{
//  public StockViewModel()
//  {
//    StockTicker = new StockTicker("MSFT", "APPL", "YHOO", "GOOG");
//    GetPriceChanges(StockTicker)
//      .GroupBy( stockArgs => stockArgs.StockSymbol )
//      .Subscribe( groupedStocks => 
//        groupedStocks.DistinctUntilChanged(stockArgs =>
//          Math.Ceiling(stockArgs.Price/10.0))
//        .ObserveOnDispatcher()
//        .Subscribe( stockArgs => 
//          StockChanges.Add( 
//            new StockChange 
//              {
//                StockSymbol = stockArgs.StockSymbol,
//                Price = stockArgs.Price
//              } ) ) );
//  }
 
//  private IObservable<PriceChangedEventArgs> GetPriceChanges(StockTicker stockTicker)
//  {
//    return Observable.FromEventPattern<PriceChangedEventArgs>(
//      eventHandler => stockTicker.OnPriceChanged += eventHandler,
//      eventHandler => stockTicker.OnPriceChanged -= eventHandler)
//        .Select(eventPattern => eventPattern.EventArgs);
//  }
 
//  private StockTicker StockTicker { get; set; }
 
//  private ObservableCollection<StockChange> _StockChanges = 
//    new ObservableCollection<StockChange>();
//  public ObservableCollection<StockChange> StockChanges
//  {
//    get { return _StockChanges; }
//    set { _StockChanges = value; }
//  }
//} 



  //  class Program
  //  {
  //      static void Main(string[] args)
  //      {

  ////          priceChangedObservable
  ////.GroupBy( stock => stock.StockSymbol )
  ////.Subscribe( group =>
  ////  group.DistinctUntilChanged( 
  ////    stock => Math.Ceiling(stock.Volume/1000.0) ) 
  ////    .Subscribe( stock => Console.WriteLine( /* ... */ ) ) )



  ////          var orderedEvents = Observable.Create<JObject>(observer =>
  ////          {
  ////              var nextVersionExpected = 1;
  ////              var previousEvents = new List<JObject>();
  ////              return events
  ////                  .ObserveOn(Scheduler.CurrentThread)
  ////                  .Subscribe(@event =>
  ////                  {
  ////                      previousEvents.Add(@event);

  ////                      var version = (long)@event["Version"];
  ////                      if (version != nextVersionExpected) return;

  ////                      foreach (var previousEvent in previousEvents.OrderBy(x => (long)x["Version"]).ToList())
  ////                      {
  ////                          if ((long)previousEvent["Version"] != nextVersionExpected)
  ////                              break;

  ////                          observer.OnNext(previousEvent);
  ////                          previousEvents.Remove(previousEvent);
  ////                          nextVersionExpected++;
  ////                      }
  ////                  });
  ////          });

  //      }
  //  }
}
