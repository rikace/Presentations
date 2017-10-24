using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Domain.Entities
{
    public class Cart : IEntity
    {
        public string CartId { get; set; }

        [ForeignKey("Product")]
        public int ProductId { get; set; }

        public int Count { get; set; }
        public DateTime DateCreated { get; set; }

        public virtual Product Product { get; set; }

        [Key]
        public int Id { get; set; }
    }
}