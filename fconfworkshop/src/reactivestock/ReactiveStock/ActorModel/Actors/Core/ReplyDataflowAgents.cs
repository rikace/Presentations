using static ReactiveStock.ActorModel.Actors.Core.OptionHelpers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Dataflow;

namespace ReactiveStock.ActorModel.Actors.Core
{
    class StatefulReplyDataflowAgent<TState, TMessage, TReply> :
                                            IReplyAgent<TMessage, TReply>
    {
        private TState state;
        private readonly ActionBlock<(TMessage,
                                      Option<TaskCompletionSource<TReply>>)> actionBlock;


        public StatefulReplyDataflowAgent(TState initialState,
                    Func<TState, TMessage, Task<TState>> projection,
                    Func<TState, TMessage, Task<(TState, TReply)>> ask,
                    CancellationTokenSource cts = null)
        {
            state = initialState;
            var options = new ExecutionDataflowBlockOptions
            {
                CancellationToken = cts != null ? cts.Token : CancellationToken.None
            };
            actionBlock = new ActionBlock<(TMessage, Option<TaskCompletionSource<TReply>>)>(
              async message => {
                  (TMessage msg, Option<TaskCompletionSource<TReply>> replyOpt) = message;
                  await replyOpt.Match(
                          none: async () => state = await projection(state, msg),
                          some: async reply => {
                              (TState newState, TReply replyresult) = await ask(state, msg);
                              state = newState;
                              reply.SetResult(replyresult);
                          });
              });
        }

        public Task<TReply> Ask(TMessage message)
        {
            var tcs = new TaskCompletionSource<TReply>();
            actionBlock.Post((message, Some(tcs)));
            return tcs.Task;
        }

        public Task Send(TMessage message) =>
            actionBlock.SendAsync((message, None));

        public void Post(TMessage message) =>
            actionBlock.Post((message, None));


        public StatefulReplyDataflowAgent(TState initialState,
                    Func<TState, TMessage, TState> projection,
                    Func<TState, TMessage, (TState, TReply)> ask,
                    CancellationTokenSource cts = null)
        {
            state = initialState;
            var options = new ExecutionDataflowBlockOptions
            {
                CancellationToken = cts != null ? cts.Token : CancellationToken.None
            };
            actionBlock = new ActionBlock<(TMessage, Option<TaskCompletionSource<TReply>>)>(
              message => {
                  (TMessage msg, Option<TaskCompletionSource<TReply>> replyOpt) = message;
                  replyOpt.Match(none: () => (state = projection(state, msg)),
                                 some: reply =>
                                 {
                                     (TState newState, TReply replyresult) = ask(state, msg);
                                     state = newState;
                                     reply.SetResult(replyresult);
                                     return state;
                                 });
              });
        }
    }
}
