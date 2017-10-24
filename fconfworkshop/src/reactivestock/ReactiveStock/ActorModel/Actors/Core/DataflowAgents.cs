using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace ReactiveStock.ActorModel.Actors.Core
{
    // TASK
    // implement a stateful agent using TPL DataFlow ActionBlock
    // The state of the Agent should be define with an initial value (seed), which can be pass as argument
    // Then, it should have an arbitrary function that transform the current state, the new message receive and return a new (or not) state
    public class StatefulDataflowAgent<TState, TMessage> : IAgent<TMessage>
    {
        private TState state;
        private readonly ActionBlock<TMessage> actionBlock;

        public StatefulDataflowAgent(
            TState initialState,
            Func<TState, TMessage, Task<TState>> action,
            CancellationTokenSource cts = null)
        {
            state = initialState;
            var token = cts?.Token ?? CancellationToken.None;
            var options = new ExecutionDataflowBlockOptions { CancellationToken = token };
            actionBlock = new ActionBlock<TMessage>(
                async msg => state = await action(state, msg), options);
        }

        public Task Send(TMessage message) => actionBlock.SendAsync(message);
        public void Post(TMessage message) => actionBlock.Post(message);


        public StatefulDataflowAgent(TState initialState, Func<TState, TMessage, TState> action, CancellationTokenSource cts = null)
        {
            state = initialState;
            var token = cts?.Token ?? CancellationToken.None;
            var options = new ExecutionDataflowBlockOptions { CancellationToken = token };
            actionBlock = new ActionBlock<TMessage>(
                msg => state = action(state, msg), options);
        }

        public TState State => state;
    }

    // TASK
    // implement a stateless agent using TPL DataFlow ActionBlock
    // The agent should have an arbitrary function that process the incoming messages
    public class StatelessDataflowAgent<TMessage> : IAgent<TMessage>
    {
        private readonly ActionBlock<TMessage> actionBlock;

        public StatelessDataflowAgent(Action<TMessage> action, CancellationTokenSource cts = null)
        {
            var token = cts?.Token ?? CancellationToken.None;
            var options = new ExecutionDataflowBlockOptions { CancellationToken = token };
            actionBlock = new ActionBlock<TMessage>(action, options);
        }

        public StatelessDataflowAgent(Func<TMessage, Task> action, CancellationTokenSource cts = null)
        {
            var token = cts?.Token ?? CancellationToken.None;
            var options = new ExecutionDataflowBlockOptions { CancellationToken = token };
            actionBlock = new ActionBlock<TMessage>(action, options);
        }

        public void Post(TMessage message) => actionBlock.Post(message);
        public Task Send(TMessage message) => actionBlock.SendAsync(message);

    }

}
