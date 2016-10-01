using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tasks.Demo
{
    public sealed class SimpleThreadPoolTaskScheduler : TaskScheduler
    {
        protected override void QueueTask(Task task)
        {
            //QueueWorkItem(() => base.TryExecuteTask(task));
        }
        protected override bool TryExecuteTaskInline(
            Task task, bool taskWasPreviouslyQueued)
        {
            return base.TryExecuteTask(task);
        }
        protected override IEnumerable<Task> GetScheduledTasks()
        {
            throw new NotSupportedException();
        }
    }

}
