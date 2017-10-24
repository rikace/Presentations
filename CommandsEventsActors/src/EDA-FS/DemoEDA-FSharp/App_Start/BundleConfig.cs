using System.Web.Optimization;

namespace DemoEDA
{
    public class BundleConfig
    {
        public static void RegisterBundles(BundleCollection bundles)
        {
            bundles.Add(new ScriptBundle("~/bundles/global").Include(
                "~/Scripts/modernizr-*",
                "~/Scripts/jquery-{version}.js",
                "~/Scripts/bootstrap*",
                "~/Scripts/knockout*",
                "~/Scripts/toastr.js"
                ));


            bundles.Add(new ScriptBundle("~/bundles/validation").Include(
                "~/Scripts/jquery.unobtrusive*",
                "~/Scripts/jquery.validate*"
                ));

            bundles.Add(new StyleBundle("~/Content/css").Include(
                "~/Content/bootstrap*",
                "~/Content/Auctions.css",
                "~/Content/Details.css",
                "~/Content/Featured.css",
                "~/Content/toastr.css"
                ));

            bundles.Add(new ScriptBundle("~/scripts/signalr").Include(
                "~/Scripts/jquery.signalR-{version}.js",
                "~/Scripts/diff_match_patch.js"));
        }
    }
}