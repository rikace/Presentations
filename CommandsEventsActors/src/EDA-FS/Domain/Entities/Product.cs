using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Domain.Entities
{
    public class Product : IEntity
    {
        private int? _categoryId;

        public Product()
        {
            Created = DateTime.Now;
        }

        [Required]
        [ForeignKey("Category")]
        public virtual int CategoryId
        {
            get
            {
                if (_categoryId == null && Category != null)
                    return Category.Id;

                return _categoryId.GetValueOrDefault();
            }
            set { _categoryId = value; }
        }

        public virtual Category Category { get; set; }

        [Required]
        [DataType(DataType.Text)]
        [StringLength(maximumLength: 500, MinimumLength = 10)]
        public string Name { get; set; }

        [DataType(DataType.MultilineText)]
        public string Description { get; set; }

        [Display(Name = "Image URL")]
        [DataType(DataType.ImageUrl)]
        public string ImageUrl { get; set; }

        [Display(Name = "Thumbnail URL")]
        [DataType(DataType.ImageUrl)]
        public string ThumbnailUrl { get; set; }

        [Display(Name = "Current Price")]
        [DataType(DataType.Currency)]
        public decimal Price { get; set; }

        [Required]
        [DataType(DataType.DateTime)]
        public DateTime Created { get; private set; }

        public virtual List<OrderDetail> OrderDetails { get; set; }

        [Key]
        public int Id { get; private set; }
    }
}