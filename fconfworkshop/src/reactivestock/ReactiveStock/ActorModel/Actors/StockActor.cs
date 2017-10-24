using System;
using System.Collections.Generic;
using ReactiveStock.ActorModel.Messages;
using ReactiveStock.ActorModel.Actors.Core;
using ReactiveStock.ExternalServices;
using System.Timers;

namespace ReactiveStock.ActorModel.Actors
{
    public static class StockActor
    {
        public static IAgent<StockAgentMessage> Create(string stockSymbol)
        {
            // Start price lookup child actor
            var gateWay = new RandomStockPriceServiceGateway();
            var priceLookupChild = StockPriceLookupActor.Create(gateWay);

            // Start Price actor
            var self =  Agent.Start<HashSet<IAgent<ChartSeriesMessage>>, StockAgentMessage>(
                new HashSet<IAgent<ChartSeriesMessage>>(), async (subscribers, message) =>
            {
                switch (message)
                {
                    case SubscribeToNewStockPricesMessage msg:
                        subscribers.Add(msg.Subscriber);
                        return subscribers;
                    case UnSubscribeFromNewStockPricesMessage msg:
                        subscribers.Remove(msg.Subscriber);
                        return subscribers;
                    case RefreshStockPriceMessage msg:
                        var request = new RefreshStockPriceMessage(msg.StockSymbol);
                        var updatedPriceMsg = await priceLookupChild.Ask(request);
                        var stockPriceMessage = new StockPriceMessage(
                            msg.StockSymbol, updatedPriceMsg.Price, updatedPriceMsg.Date);
                        foreach (var subscriber in subscribers)
                        {
                            subscriber.Post(stockPriceMessage);
                        }
                        return subscribers;
                    default:
                        throw new ArgumentException(
                           message: "message is not a recognized",
                           paramName: nameof(message));
                }
            });

            // Start timer that trigger refresh events (just to avoid dependency on Rx)
            Timer timer = new Timer(750);
            var refreshMsg = new RefreshStockPriceMessage(stockSymbol);
            timer.AutoReset = true;
            timer.Elapsed += (sender, e) => self.Post(refreshMsg);
            timer.Start();
            // TODO: Ideally we have to stop and dispose timer

            return self;
        }
    }
}
