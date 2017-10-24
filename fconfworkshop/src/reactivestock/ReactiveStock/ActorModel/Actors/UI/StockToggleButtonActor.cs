using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ReactiveStock.ActorModel.Messages;
using ReactiveStock.ViewModel;
using ReactiveStock.ActorModel.Actors.Core;

namespace ReactiveStock.ActorModel.Actors.UI
{
    public static class StockToggleButtonActor
    {
        public static IAgent<FlipToggleMessage> Create(
            IAgent<StocksCoordinatorMessage> coordinatorActor,
            StockToggleButtonViewModel viewModel,
            string stockSymbol)
        {
            return Agent.Start<bool, FlipToggleMessage>(false, (isToggledOn, message) =>
            {
                if (isToggledOn) {
                    coordinatorActor.Post(new UnWatchStockMessage(stockSymbol));
                    viewModel.UpdateButtonTextToOff();
                } else {
                    coordinatorActor.Post(new WatchStockMessage(stockSymbol));
                    viewModel.UpdateButtonTextToOn();
                }
                return !isToggledOn;
            });
        }
    }
}
