namespace ReactiveStock.ExternalServices
{
    public interface IStockPriceServiceGateway
    {
        decimal GetLatestPrice(string stockSymbol);
    }
}
