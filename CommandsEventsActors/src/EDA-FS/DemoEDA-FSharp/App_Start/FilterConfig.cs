using System.Web.Mvc;
using DemoEDA.Filters;

namespace DemoEDA
{
    public class FilterConfig
    {
        public static void RegisterGlobalFilters(GlobalFilterCollection filters)
        {
            filters.Add(new CategoriesActionFilter());
            filters.Add(new HandleErrorAttribute());
        }
    }
}