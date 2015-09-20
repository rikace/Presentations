namespace Easj360FSharp

open System.Net
open System.IO


type StockAnalyzer(ticker, lprices, days) =
    let prices = lprices
                    |> Seq.map snd // get prices only from (date, price) tuple
                    |> Seq.take days

    static member private loadPrices ticker =
        let url = "http://ichart.finance.yahoo.com/table.csv?s=" + ticker + "&a=05&b=11&c=1998&d=04&e=20&f=2011&g=d&ignore=.csv"
        let req = WebRequest.Create(url)
        use response = req.GetResponse() // "use" keyword calls Dispose() method when it goes out of scope
        use stream = response.GetResponseStream()
        use reader = new StreamReader(stream)
        let csv = reader.ReadToEnd()

        let prices =
            csv.Split([|'\n'|])
            |> Seq.skip 1
            |> Seq.map (fun currLine -> currLine.Split([|','|]))
            |> Seq.filter (fun values -> (values |> Seq.length = 7))
            |> Seq.map (fun values ->
                System.DateTime.Parse(values.[0]),
                (float) values.[6]
            )
        (ticker, prices) (* return the ticker also because we want to display it in results *)

    static member GetAnalyzers(tickers, days) =
        tickers
        |> Seq.map StockAnalyzer.loadPrices
        |> Seq.map (fun (ticker, prices) -> new StockAnalyzer(ticker, prices, days))




    member s.Ticker = ticker

    member s.Return =
        let lastPrice = prices |> Seq.nth 0
        let startPrice = prices |> Seq.nth (days - 1)
        lastPrice / startPrice

    member s.StdDev =
        let logReturns =
            prices
            |> Seq.pairwise
            |> Seq.map (fun (x,y) -> log (x/y))
        let mean = logReturns |> Seq.average
        let square x = x * x
        let variance = logReturns |> Seq.averageBy (fun r -> square (r - mean))

        sqrt(variance) // return Standard Deviation

type StockAnalyzerAsync(ticker, lprices, days) =
    let prices = lprices
                    |> Seq.map snd // get prices only from (date, price) tuple
                    |> Seq.take days

    static member private loadPrices ticker = async {
        let url = "http://ichart.finance.yahoo.com/table.csv?s=" + ticker + "&a=05&b=11&c=1998&d=04&e=20&f=2011&g=d&ignore=.csv"
        let req = WebRequest.Create(url)
        use! response = req.AsyncGetResponse() // "use!" keyword calls Dispose() method when it goes out of scope
        use stream = response.GetResponseStream()
        use reader = new StreamReader(stream)
        let csv = reader.ReadToEnd()

        let prices =
            csv.Split([|'\n'|])
            |> Seq.skip 1
            |> Seq.map (fun currLine -> currLine.Split([|','|]))
            |> Seq.filter (fun values -> (values |> Seq.length = 7))
            |> Seq.map (fun values ->
                System.DateTime.Parse(values.[0]),
                (float) values.[6]
            )
        return (ticker, prices) (* return the ticker also because we want to display it in results *) }

    static member GetAnalyzers(tickers, days) =
        tickers
        |> Seq.map StockAnalyzerAsync.loadPrices
        |> Async.Parallel          // **** run Async operations in parallel
        |> Async.RunSynchronously  // **** Start async operations AND synchronously wait for all of them to complete (i.e., "merge" back async operations back into the main thread)
        |> Seq.map (fun (ticker, prices) -> new StockAnalyzer(ticker, prices, days))


    member s.Ticker = ticker

    member s.Return =
        let lastPrice = prices |> Seq.nth 0
        let startPrice = prices |> Seq.nth (days - 1)
        lastPrice / startPrice

    member s.StdDev =
        let logReturns =
            prices
            |> Seq.pairwise
            |> Seq.map (fun (x,y) -> log (x/y))
        let mean = logReturns |> Seq.average
        let square x = x * x
        let variance = logReturns |> Seq.averageBy (fun r -> square (r - mean))

        sqrt(variance) // return Standard Deviation


(*var tickers = new[] { "MSFT", "ORCL", "EBAY", "MSTR", "GOOG", "REVU" };


			//------------------------------------------------------------------------------
			// Non-Async version
			//------------------------------------------------------------------------------
			var analyzers = StockAnalyzer.GetAnalyzers(tickers, 1500 /* days */);


			//------------------------------------------------------------------------------
			// Async version
			//------------------------------------------------------------------------------
			//var analyzers = StockAnalyzerAsync.GetAnalyzers(tickers, 365 /* days */);


			foreach (var a in analyzers)
			{
				Console.WriteLine("========\n| {0} |\n========", a.Ticker);
				Console.WriteLine("Return: {0}\nStdDev: {1}\n\n", a.Return, a.StdDev);
			}*)