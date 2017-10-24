using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace DataFlowAgent
{
    public class StatefulDataflowAgent<TState, TMessage> 
    {
        private TState state;
        private readonly ActionBlock<TMessage> actionBlock;

        public static StatefulDataflowAgent<TState, TMessage> Start(TState initialState,
            Func<TState, TMessage, Task<TState>> action,
            CancellationTokenSource cts = null) => new StatefulDataflowAgent<TState, TMessage>(initialState, action, cts);
        public StatefulDataflowAgent(
            TState initialState,
            Func<TState, TMessage, Task<TState>> action,  
            CancellationTokenSource cts = null)
        {
            state = initialState;
            var options = new ExecutionDataflowBlockOptions
            {
                CancellationToken = cts != null ?
                    cts.Token : CancellationToken.None  // #B
            };
            actionBlock = new ActionBlock<TMessage>(    // #C
                async msg => state = await action(state, msg), options);
        }

        public Task Send(TMessage message) => actionBlock.SendAsync(message);
        public void Post(TMessage message) => actionBlock.Post(message);


        public StatefulDataflowAgent(TState initialState, Func<TState, TMessage, TState> action, CancellationTokenSource cts = null)
        {
            state = initialState;
            var options = new ExecutionDataflowBlockOptions
            {
                CancellationToken = cts != null ? cts.Token : CancellationToken.None
            };
            actionBlock = new ActionBlock<TMessage>(
                msg => state = action(state, msg), options);
        }

        public TState State => state;
    }


    public class StatelessDataflowAgent<TMessage>  
    {
        private readonly ActionBlock<TMessage> actionBlock;

        public static StatelessDataflowAgent<TMessage> Start(Action<TMessage> action, CancellationTokenSource cts = null) => new StatelessDataflowAgent<TMessage>(action, cts);
        public StatelessDataflowAgent(Action<TMessage> action, CancellationTokenSource cts = null)
        {
            var options = new ExecutionDataflowBlockOptions
            {
                CancellationToken = cts != null ? cts.Token : CancellationToken.None
            };
            actionBlock = new ActionBlock<TMessage>(action, options);
        }

        public StatelessDataflowAgent(Func<TMessage, Task> action, CancellationTokenSource cts = null)
        {
            var options = new ExecutionDataflowBlockOptions
            {
                CancellationToken = cts == null ? cts.Token : CancellationToken.None
            };
            actionBlock = new ActionBlock<TMessage>(action, options);
        }

        public void Post(TMessage message) => actionBlock.Post(message);
        public Task Send(TMessage message) => actionBlock.SendAsync(message);

    }
}
