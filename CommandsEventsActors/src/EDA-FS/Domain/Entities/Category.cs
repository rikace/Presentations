using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Domain.Entities
{
    public class Category : IEntity
    {
        private string _key;

        [Required]
        [DataType(DataType.Text)]
        [StringLength(maximumLength: 20, MinimumLength = 3)]
        public string Key
        {
            get { return _key = (_key ?? Name.ToLower().Replace(" ", "_")); }
            set { _key = value; }
        }

        [Required]
        [DataType(DataType.Text)]
        [StringLength(maximumLength: 30, MinimumLength = 3)]
        public string Name { get; set; }


        public virtual ICollection<Product> Products { get; set; }

        [Key]
        public int Id { get; private set; }
    }
}