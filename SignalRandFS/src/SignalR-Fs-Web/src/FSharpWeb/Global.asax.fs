namespace FSharpWeb

open System
open System.Net.Http
open System.Web
open System.Web.Http
open System.Web.Mvc
open System.Web.Routing
open System.Web.Optimization

type BundleConfig() =
    static member RegisterBundles (bundles:BundleCollection) =
        bundles.Add(ScriptBundle("~/bundles/jquery").Include([|"~/Scripts/jquery-{version}.js"|]))

        // Use the development version of Modernizr to develop with and learn from. Then, when you're
        // ready for production, use the build tool at http://modernizr.com to pick only the tests you need.
        bundles.Add(ScriptBundle("~/bundles/modernizr").Include([|"~/Scripts/modernizr-*"|]))

        bundles.Add(ScriptBundle("~/bundles/bootstrap").Include(
                        "~/Scripts/bootstrap.js",
                        "~/Scripts/respond.js"))

        bundles.Add(StyleBundle("~/Content/css").Include(
                        "~/Content/bootstrap.css",
                        "~/Content/site.css"))

/// Route for ASP.NET MVC applications
type Route = { 
    controller : string
    action : string
    id : UrlParameter }

type HttpRoute = {
    controller : string
    id : RouteParameter }


type internal AsyncCallableHandler(messageHandler) =
    inherit DelegatingHandler(messageHandler)
    member internal x.CallSendAsync(request, cancellationToken) =
        base.SendAsync(request, cancellationToken)

(*  the Global.fs file contains most of the code.
    The Route type is an F# record type that is being used to contain routing definitions. 
    Record types are immutable by default. Because of this, they go very well with the highly 
    concurrent and stateless nature of the Web *)
type Global() =
    inherit System.Web.HttpApplication() 

//    let loggingHandler =
//        { new DelegatingHandler() with
//            member x.SendAsync(request, cancellationToken) =
//                let wrapped = new AsyncCallableHandler(base.InnerHandler)
//                let workflow = async {
//                    let! requestContent =
//                        request.Content.ReadAsStringAsync()
//                        |> Async.AwaitTask
//                    log requestContent
//                    let! response =
//                        wrapped.SendAsync(request, cancellationToken)
//                        |> Async.AwaitTask
//                    let! responseContent =
//                        response.Content.ReadAsStringAsync()
//                        |> Async.AwaitTask
//                    log responseContent
//                    return response
//                }
//                Async.StartAsTask(workflow, cancellationToken = cancellationToken) }

    static member RegisterWebApi(config: HttpConfiguration) =
        // Configure routing
        config.MapHttpAttributeRoutes()
        config.Routes.MapHttpRoute(
            "DefaultApi", // Route name
            "api/{controller}/{id}", // URL with parameters
            { controller = "{controller}"; id = RouteParameter.Optional } // Parameter defaults
        ) |> ignore

        // Configure serialization
        config.Formatters.XmlFormatter.UseXmlSerializer <- true
        config.Formatters.JsonFormatter.SerializerSettings.ContractResolver <- Newtonsoft.Json.Serialization.CamelCasePropertyNamesContractResolver()

        // Additional Web API settings

    static member RegisterFilters(filters: GlobalFilterCollection) =
        filters.Add(new HandleErrorAttribute())

    static member RegisterRoutes(routes:RouteCollection) =
        routes.IgnoreRoute("{resource}.axd/{*pathInfo}")
        routes.MapRoute(
            "Default", // Route name
            "{controller}/{action}/{id}", // URL with parameters
            { controller = "Home"; action = "Index"; id = UrlParameter.Optional } // Parameter defaults
        ) |> ignore

    member x.Application_Start() =
        AreaRegistration.RegisterAllAreas()
        GlobalConfiguration.Configure(Action<_> Global.RegisterWebApi)
        Global.RegisterFilters(GlobalFilters.Filters)
        Global.RegisterRoutes(RouteTable.Routes)
        BundleConfig.RegisterBundles BundleTable.Bundles