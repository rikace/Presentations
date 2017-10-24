using ReactiveStock.ActorModel.Actors.Core;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ReactiveStock.ActorModel.Messages
{
    public class SubscribeToNewStockPricesMessage : StockAgentMessage
    {
        public IAgent<ChartSeriesMessage> Subscriber { get; private set; }

        public SubscribeToNewStockPricesMessage(IAgent<ChartSeriesMessage> subscribingActor)
        {
            Subscriber = subscribingActor;
        }
    }
}
