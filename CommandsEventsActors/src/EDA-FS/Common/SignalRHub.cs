using Microsoft.AspNet.SignalR;

namespace DemoEDAFSharp.Infrastructure
{
    public class SignalRHub : Hub
    {
        public void Send(string message)
        {
            Clients.All.broadcastMessage(message);
        }
    }
}