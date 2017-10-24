using System;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Linq;
using System.Web.Mvc;
using DemoEDA.Models;
using DemoEDAFSharp.Commands;
using DemoEDAFSharp.Infrastructure;
using DemoEDAFSharp.Infrastructure.Cache;
using DemoEDAFSharp.Models;
using Domain.DataAccess;
using Domain.Entities;

namespace DemoEDAFSharp.Controllers
{
    [Export]
    [PartCreationPolicy(CreationPolicy.NonShared)]
    public class StoreEDAController : Controller
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

            string cartId = ShoppingCart.GetCartId(HttpContext);

            ServiceLocator.CommandBus.Dispatch(new AddProductCommand(cartId, item, quantity));

            return RedirectToAction("Index", "StoreEDA");
        }

        [HttpPost]
        public ActionResult Remove(int id)
        {
            Product item = storetUnitOfWork.Products.FindById(id);
            if (item == null)
                return HttpNotFound("Item not found");

            string cartId = ShoppingCart.GetCartId(HttpContext);

            ServiceLocator.CommandBus.Dispatch(new RemoveProductCommand(cartId, item));

            return RedirectToAction("Index", "StoreEDA");
        }


        [HttpPost]
        public ActionResult CheckOut(Order model)
        {
            var order = new Order();
            if (TryUpdateModel(order))
            {
                string cartId = ShoppingCart.GetCartId(HttpContext);

                ServiceLocator.CommandBus.Dispatch(new SubmitOrderCommand(cartId, order));

                return RedirectToAction("Complete", "StoreEDA", new {id = order.Id});
            }
            return View();
        }

        public ActionResult Complete(int id)
        {
            Order order = storetUnitOfWork.Order.FindById(id);
            ViewBag.Message =
                String.Format(
                    "Thank you for you order {0} {1}, your order number is {2}." +
                    "You will receive a notification by email with updates",
                    order.FirstName, order.LastName, order.Id);

            string cartId = ShoppingCart.GetCartId(HttpContext);
            ServiceLocator.CommandBus.Dispatch(new EmptyCardCommand(cartId));

            return View("Complete");
        }

        [HttpPost]
        public ActionResult EmptyCart()
        {
            string cartId = ShoppingCart.GetCartId(HttpContext);
            ServiceLocator.CommandBus.Dispatch(new EmptyCardCommand(cartId));

            return View("Index");
        }

        [ChildActionOnly]
        public ActionResult Products()
        {
            List<Product> items = Cache.Instance.GetOrSet("Products", () => storetUnitOfWork.Products.FindAll().ToList(),
                TimeSpan.FromMinutes(1));
            IEnumerable<ProductViewModel> viewModels = items.Select(Mapper.Map<Product, ProductViewModel>);
            return PartialView("_ProductList", viewModels);
        }

        public ActionResult Details(int id)
        {
            Product item = storetUnitOfWork.Products.FindById(id);

            if (item == null)
                HttpNotFound("Item not found");
            ProductViewModel viewModel = Mapper.Map<Product, ProductViewModel>(item);
            return View("Details", viewModel);
        }

        public ActionResult CheckOut()
        {
            Order fakeOrder = FakeOrderDetailsFactory.CreateFakeOrderDetails();
            return View("CheckOut", fakeOrder);
        }
    }
}