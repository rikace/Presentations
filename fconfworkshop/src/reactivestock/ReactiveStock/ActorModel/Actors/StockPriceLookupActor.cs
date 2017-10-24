using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ReactiveStock.ActorModel.Messages;
using ReactiveStock.ExternalServices;
using ReactiveStock.ActorModel.Actors.Core;

namespace ReactiveStock.ActorModel.Actors
{
    public static class StockPriceLookupActor
    {
        public static IReplyAgent<RefreshStockPriceMessage, UpdatedStockPriceMessage>
            Create(IStockPriceServiceGateway stockPriceServiceGateway)
        {
            return Agent.Start<RefreshStockPriceMessage, UpdatedStockPriceMessage>(message =>
            {
                var latestPrice = stockPriceServiceGateway.GetLatestPrice(message.StockSymbol);
                return new UpdatedStockPriceMessage(latestPrice, DateTime.Now);
            });
        }
    }
}
