using System.Web.Mvc;
using DemoEDA.Infrastructure;

namespace DemoEDAFSharp.Controllers
{
    public class HomeController : Controller
    {
        public ActionResult Index()
        {
            HttpContext.Session.RemoveAll();
            return View();
        }
    }
}