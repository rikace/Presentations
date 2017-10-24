using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReactiveStock.ActorModel.Messages
{
    class UnWatchStockMessage : StocksCoordinatorMessage
    {
        public string StockSymbol { get; private set; }

        public UnWatchStockMessage(string stockSymbol)
        {
            StockSymbol = stockSymbol;
        }
    }
}
