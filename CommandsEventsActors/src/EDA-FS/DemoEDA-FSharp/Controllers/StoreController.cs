using System;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Linq;
using System.Web.Mvc;
using DemoEDA.Models;
using DemoEDAFSharp.Infrastructure;
using DemoEDAFSharp.Infrastructure.Cache;
using DemoEDAFSharp.Models;
using Domain.DataAccess;
using Domain.Entities;

namespace DemoEDAFSharp.Controllers
{
    [Export]
    [PartCreationPolicy(CreationPolicy.NonShared)]
    public class StoreController : Controller
    {
        [Import] private IStoretUnitOfWork storetUnitOfWork;

        public ActionResult Index()
        {
            ShoppingCart cart = ShoppingCart.GetCart(HttpContext, storetUnitOfWork);

            var viewModel = new ShoppingCartViewModel
            {
                CartItems = cart.GetCartItems(),
                CartTotal = cart.GetTotal()
            };
            return View("Index", viewModel);
        }

        [HttpPost]
        public ActionResult AddToCart(int id, int quantity = 1)
        {
            Product item = storetUnitOfWork.Products.FindById(id);

            if (item == null)
                return HttpNotFound("Item not found");

            ShoppingCart cart = ShoppingCart.GetCart(HttpContext, storetUnitOfWork);

            cart.AddToCart(item, quantity);

            return RedirectToAction("Index", "Store");
        }

        public ActionResult Details(int id)
        {
            Product item = storetUnitOfWork.Products.FindById(id);

            if (item == null)
                HttpNotFound("Item not found");

            ProductViewModel viewModel = Mapper.Map<Product, ProductViewModel>(item);

            return View("Details", viewModel);
        }

        [HttpPost]
        public ActionResult Remove(int id)
        {
            ShoppingCart cart = ShoppingCart.GetCart(HttpContext, storetUnitOfWork);
            cart.RemoveFromCart(id);

            return RedirectToAction("Index", "Store");
        }

        [ChildActionOnly]
        public ActionResult Products()
        {
            List<Product> items = Cache.Instance.GetOrSet("Products", () => storetUnitOfWork.Products.FindAll().ToList(),
                TimeSpan.FromMinutes(1));
            IEnumerable<ProductViewModel> viewModels = items.Select(Mapper.Map<Product, ProductViewModel>);
            return PartialView("_ProductList", viewModels);
        }

        public ActionResult CheckOut()
        {
            Order fakeOrder = FakeOrderDetailsFactory.CreateFakeOrderDetails();
            return View("CheckOut", fakeOrder);
        }

        [HttpPost]
        public ActionResult CheckOut(Order model)
        {
            var order = new Order();
            if (TryUpdateModel(order))
            {
                order.OrderDate = DateTime.Now;
                storetUnitOfWork.Order.Add(order);
                storetUnitOfWork.Commit();

                ShoppingCart cart = ShoppingCart.GetCart(HttpContext, storetUnitOfWork);
                cart.CreateOrder(order);

                return RedirectToAction("Complete", "Store", new {id = order.Id});
            }
            return View();
        }

        [HttpPost]
        public ActionResult EmptyCart()
        {
            ShoppingCart cart = ShoppingCart.GetCart(HttpContext, storetUnitOfWork);
            cart.EmptyCart();

            return View("Index");
        }

        public ActionResult Complete(int id)
        {
            Order order = storetUnitOfWork.Order.FindById(id);
            ViewBag.Message = String.Format("Thank you for you order {0} {1}", order.FirstName, order.LastName);
            ShoppingCart cart = ShoppingCart.GetCart(HttpContext, storetUnitOfWork);
            cart.EmptyCart();

            return View("Complete");
        }
    }
}