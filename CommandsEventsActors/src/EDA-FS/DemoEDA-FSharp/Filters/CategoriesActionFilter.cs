using System;
using System.ComponentModel.Composition;
using System.Linq;
using System.Web.Mvc;
using DemoEDAFSharp.Infrastructure.Cache;
using Domain.DataAccess;
using Domain.Entities;

namespace DemoEDA.Filters
{
    [Export]
    [PartCreationPolicy(CreationPolicy.NonShared)]
    public class CategoriesActionFilter : ActionFilterAttribute
    {
        [Import] private IStoretUnitOfWork _repository;

        public CategoriesActionFilter()
        {
            _repository = (DependencyResolver.Current as MefDependencyResolver).GetService<IStoretUnitOfWork>();
        }

        public override void OnActionExecuting(ActionExecutingContext filterContext)
        {
            Category[] categories = Cache.Instance.GetOrSet("Categories",
                () => _repository.Categories.FindAll().ToArray(),
                TimeSpan.FromMinutes(1));

            filterContext.Controller.ViewBag.Categories = categories;
        }
    }
}