using System.Collections.Generic;
using Domain.Entities;

namespace DemoEDAFSharp.Models
{
    public class ShoppingCartViewModel
    {
        public List<Cart> CartItems { get; set; }
        public decimal CartTotal { get; set; }
    }
}

//public class Cart
//{
//    [Key]
//    public int RecordId { get; set; }
//    public string CartId { get; set; }
//    public int ProductId { get; set; }
//    public int Count { get; set; }
//    public System.DateTime DateCreated { get; set; }

//    public virtual Album Album { get; set; }
//}