using System.ComponentModel.Composition;
using System.Data.Entity;
using Domain.Entities;

namespace Domain.DataAccess
{
    [Export(typeof (DbContext))]
    public class StoreContext : DbContext
    {
        public StoreContext()
            : base("name=StoreContext")
        {
        }

        public DbSet<Product> Products { get; set; }
        public DbSet<Category> Categories { get; set; }
        public DbSet<Cart> Carts { get; set; }
        public DbSet<Order> Orders { get; set; }
        public DbSet<OrderDetail> OrderDetails { get; set; }
    }
}