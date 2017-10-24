namespace DemoEDA.Models
{
    public class ProductViewModel
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        public string ImageUrl { get; set; }
        public string ThumbnailUrl { get; set; }
        public decimal Price { get; set; }
        public long CategoryId { get; set; }

        public string PriceDisplay
        {
            get { return Price.ToString("c"); }
        }
    }
}