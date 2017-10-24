using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using GalaSoft.MvvmLight;
using GalaSoft.MvvmLight.Command;
using ReactiveStock.ActorModel;
using ReactiveStock.ActorModel.Actors.UI;
using ReactiveStock.ActorModel.Messages;
using ReactiveStock.ActorModel.Actors.Core;

namespace ReactiveStock.ViewModel
{
    public class StockToggleButtonViewModel : ViewModelBase
    {
        private string _buttonText;

        public string StockSymbol { get; set; }

        public ICommand ToggleCommand { get; set; }

        public IAgent<FlipToggleMessage> StockToggleButtonActorRef { get; private set; }

        public string ButtonText
        {
            get { return _buttonText; }
            set { Set(() => ButtonText, ref _buttonText, value); }
        }

        public StockToggleButtonViewModel(IAgent<StocksCoordinatorMessage> stocksCoordinatorRef, string stockSymbol)
        {
            StockSymbol = stockSymbol;

            StockToggleButtonActorRef =
                StockToggleButtonActor.Create(stocksCoordinatorRef, this, stockSymbol);

            ToggleCommand = new RelayCommand(
                () => StockToggleButtonActorRef.Post(new FlipToggleMessage()));

            UpdateButtonTextToOff();
        }

        public void UpdateButtonTextToOff()
        {
            ButtonText = ConstructButtonText(false);
        }

        public void UpdateButtonTextToOn()
        {
            ButtonText = ConstructButtonText(true);
        }

        private string ConstructButtonText(bool isToggledOn)
        {
            return string.Format("{0} {1}",
                StockSymbol,
                isToggledOn ? "(on)" : "(off)");
        }
    }
}
