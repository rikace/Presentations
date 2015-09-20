namespace Utility

//#light

open System;
open System.Net
open System.IO
open System.Text
open Microsoft.FSharp.Control.WebExtensions

//#r "FSharp.PowerPack.dll"

module srcModuleWord = 

    let url = "http://ichart.finance.yahoo.com/table.csv?s=MSFT&a=4&b=1&c=2000&d=12&e=12&f=12&g=d&ignore=.csv";


    let RetrieveDateInfo (date:DateTime) =
      (date.Day, date.Month-1, date.Year)

    let CreateUrl symbol startDate endDate =
      let startDay, startMonth, startYear = RetrieveDateInfo startDate
      let endDay, endMonth, endYear = RetrieveDateInfo endDate
      let query = String.Format("&a={0}&b={1}&c={2}&d={3}&e={4}&f={5}&g=d&ignore=.csv", startMonth, startDay, startYear, endMonth, endDay, endYear)  
      let url = "http://ichart.finance.yahoo.com/table.csv?s=" + symbol + query
      url
  
    let parseData (data:string) =
        data.Split([|'\n'|])
        |>  Seq.map (fun f -> f.Split([|','|]))
        |>  Seq.skip 1
        |>  Seq.filter (fun f -> f.Length = 7)
        |>  Seq.map (fun f -> (DateTime.Parse(f.[0]), f.[6]))
        //(ticker, prices)

    let ReadQuotesAsync symbol date1 date2 =
          async {
            let url = CreateUrl symbol date1 date2
            let request = WebRequest.Create(url)
            use! response = request.AsyncGetResponse()
            use reader = new StreamReader(response.GetResponseStream(), Encoding.ASCII)
            let! data = reader.AsyncReadToEnd()
            return parseData data
            }

    let task1 = ReadQuotesAsync "MSFT" (DateTime.Parse("2000/12/01")) (DateTime.Parse("2005/12/01"))
    let task2 = ReadQuotesAsync "EBAY" (DateTime.Parse("2000/12/01")) (DateTime.Parse("2005/12/01"))
    let task3 = ReadQuotesAsync "AMZN" (DateTime.Parse("2000/12/01")) (DateTime.Parse("2005/12/01"))
    let task4 = ReadQuotesAsync "EMC" (DateTime.Parse("2000/12/01")) (DateTime.Parse("2005/12/01"))

    let getQuotes = [| task1; task2; task3; task4 |]
                    |> Async.Parallel
                    |> Async.RunSynchronously

    //////////////////////

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

    ////////////////////////////////

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