using System;
using Domain.Entities;

namespace DemoEDAFSharp.Models
{
    public class FakeOrderDetailsFactory
    {
        public static Order CreateFakeOrderDetails()
        {
            return new Order
            {
                Address = "Somewhere Rd",
                City = "Hall",
                Country = "ASU",
                Email = "myfake@email.not",
                FirstName = "Riccardo",
                LastName = "Terrell",
                OrderDate = DateTime.Now,
                Phone = "555-555-5551",
                State = "DM",
                PostalCode = "54321"
            };
        }
    }
}