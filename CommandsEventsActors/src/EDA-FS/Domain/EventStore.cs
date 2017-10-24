using System;
using System.Collections.Generic;

namespace Domain
{
    public interface IEventStore<TEvent> where TEvent : class
    {
        void SaveEvent(Guid id, TEvent @event);
        void SaveEvents(Guid id, IEnumerable<TEvent> eventsn);
        List<TEvent> GetEvents(Guid id);
    }
}