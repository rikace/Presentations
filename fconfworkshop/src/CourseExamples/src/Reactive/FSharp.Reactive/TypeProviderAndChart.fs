module TypeProviderAndChart

open FSharp.Data
open FSharp.Charting

let wb = WorldBankData.GetDataContext()

let birthRate = wb.Countries.Israel.Indicators.``Birth rate, crude (per 1,000 people)``.Values
let mobileSubscription = wb.Countries.Israel.Indicators.``Mobile cellular subscriptions (per 100 people)``.Values

Chart.Combine(
   [ Chart.Line(birthRate,Name="Birth rate")
     Chart.Line(mobileSubscription,Name="Mobile subscriptions") ])
|> Chart.WithLegend true
|> Chart.Show