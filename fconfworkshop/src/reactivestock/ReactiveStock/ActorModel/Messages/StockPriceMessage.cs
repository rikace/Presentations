using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReactiveStock.ActorModel.Messages
{
    public class StockPriceMessage : ChartSeriesMessage
    {
        public string StockSymbol { get; private set; }
        public decimal StockPrice { get; private set; }
        public DateTime Date { get; private set; }

        public StockPriceMessage(string stockSymbol, decimal stockPrice, DateTime date)
        {
            StockSymbol = stockSymbol;
            StockPrice = stockPrice;
            Date = date;
        }
    }
}
