using System;
using System.Collections.Generic;
using System.Linq;
using Common.Framework;
using DemoEDAFSharp.Events;
using Domain;

namespace DemoEDAFSharp.Framework.Events
{
    public class EventStore : IEventStore<Event>
    {
        private readonly Dictionary<Guid, List<EventDescriptor>> _current =
            new Dictionary<Guid, List<EventDescriptor>>();

        private readonly IEventPublisher _publisher;

        public EventStore(IEventPublisher publisher)
        {
            _publisher = publisher;
        }

        public void SaveEvent(Guid id, Event @event)
        {
            SaveEvents(id, new List<Event> {@event});
        }

        public void SaveEvents(Guid aggregateId, IEnumerable<Event> events)
        {
            List<EventDescriptor> eventDescriptors;
            if (!_current.TryGetValue(aggregateId, out eventDescriptors))
            {
                eventDescriptors = new List<EventDescriptor>();
                _current.Add(aggregateId, eventDescriptors);
            }
            foreach (Event @event in events)
            {
                eventDescriptors.Add(new EventDescriptor(aggregateId, @event));
                _publisher.Publish(@event);
            }
        }

        public List<Event> GetEvents(Guid id)
        {
            List<EventDescriptor> eventDescriptors;
            if (!_current.TryGetValue(id, out eventDescriptors))
            {
                throw new ArgumentNullException("");
            }
            return eventDescriptors.Select(desc => desc.EventData).ToList();
        }

        private struct EventDescriptor
        {
            public readonly Event EventData;
            private readonly Guid Id;

            public EventDescriptor(Guid id, Event eventData)
            {
                EventData = eventData;
                this.Id = id;
            }
        }
    }
}