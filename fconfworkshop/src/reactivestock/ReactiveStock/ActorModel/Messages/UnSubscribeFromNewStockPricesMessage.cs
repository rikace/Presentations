using ReactiveStock.ActorModel.Actors.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReactiveStock.ActorModel.Messages
{
    public class UnSubscribeFromNewStockPricesMessage : StockAgentMessage
    {
        public IAgent<ChartSeriesMessage> Subscriber { get; private set; }

        public UnSubscribeFromNewStockPricesMessage(IAgent<ChartSeriesMessage> unsubscribingActor)
        {
            Subscriber = unsubscribingActor;
        }
    }
}
