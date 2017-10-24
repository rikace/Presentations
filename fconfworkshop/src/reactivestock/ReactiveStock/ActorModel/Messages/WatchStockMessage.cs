using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReactiveStock.ActorModel.Messages
{
    class WatchStockMessage : StocksCoordinatorMessage
    {
        public string StockSymbol { get; private set; }

        public WatchStockMessage(string stockSymbol)
        {
            StockSymbol = stockSymbol;
        }

    }
}
