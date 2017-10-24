using System;
using Common.Framework;

namespace DemoEDAFSharp.Events
{
    // Events are the result of some action already happened in the Domain
    // Evenst cannot be declined, they are evidence
    // in DDD Commands are componentes of the Ibiquitous Language used to describe the domain
    public class Event : IEvent
    {
        public Guid Id { get; protected set; }
    }
}