using System;
using Common.Framework;

namespace DemoEDAFSharp.Commands
{
    // Command are directives to the domain to perfomr some action
    // Commands can be rejected by the domain
    // Processing a command will result in 0:n EVnets raised
    // in DDD Commands are componentes of the Ibiquitous Language used to describe the domain
    public class Command : ICommand
    {
        public Guid Id { get; protected set; }
    }
}