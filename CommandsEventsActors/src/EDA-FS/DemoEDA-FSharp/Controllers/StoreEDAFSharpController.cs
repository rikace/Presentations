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
using EDAFSharp;

namespace DemoEDAFSharp.Controllers
{
    [Export]
    [PartCreationPolicy(CreationPolicy.NonShared)]
    public class StoreEDAFSharpController : Controller
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

            ServiceLocator.FSharpBus.DispacthCommand(MessageBus.Command.NewAddProductCommand(new Guid(cartId), item,
                quantity));

            return RedirectToAction("Index", "StoreEDAFSharp");
        }

        [HttpPost]
        public ActionResult Remove(int id)
        {
            Product item = storetUnitOfWork.Products.FindById(id);
            if (item == null)
                return HttpNotFound("Item not found");

            string cartId = ShoppingCart.GetCartId(HttpContext);

            ServiceLocator.FSharpBus.DispacthCommand(MessageBus.Command.NewRemoveProductCommand(new Guid(cartId), item));

            return RedirectToAction("Index", "StoreEDAFSharp");
        }


        [HttpPost]
        public ActionResult CheckOut(Order model)
        {
            var order = new Order();
            if (TryUpdateModel(order))
            {
                string cartId = ShoppingCart.GetCartId(HttpContext);

                ServiceLocator.FSharpBus.DispacthCommand(MessageBus.Command.NewSubmitOrderCommand(new Guid(cartId),
                    order));

                return RedirectToAction("Complete", "StoreEDAFSharp", new {id = order.Id});
            }
            return View();
        }

        public ActionResult Complete(int id)
        {
            ViewBag.Message = "Thank you for you order, you will receive a notification by email with updates";

            string cartId = ShoppingCart.GetCartId(HttpContext);
            ServiceLocator.FSharpBus.DispacthCommand(
                MessageBus.Command.NewEmptyCardCommand(new Guid(cartId)));

            return View("Complete");
        }

        [HttpPost]
        public ActionResult EmptyCart()
        {
            string cartId = ShoppingCart.GetCartId(HttpContext);
            ServiceLocator.FSharpBus.DispacthCommand(MessageBus.Command.NewEmptyCardCommand(new Guid(cartId)));

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