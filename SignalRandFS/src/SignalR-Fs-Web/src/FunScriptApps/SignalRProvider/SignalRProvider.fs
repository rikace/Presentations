module SignalRProvider

open System.IO
open ProviderImplementation.ProvidedTypes

open Microsoft.FSharp.Core.CompilerServices
open System.Reflection
open System
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns

open FunScript.TypeScript
open Microsoft.AspNet.SignalR.Hubs
open Microsoft.AspNet.SignalR

open ReflectionProxy
open SignalRProviderRuntime

let getTypes assemblies = 
    let appDomain = AppDomain.CreateDomain("signalRprovider", null, new AppDomainSetup(ShadowCopyFiles="false",DisallowApplicationBaseProbing=true))
    let dll = typeof<ReflectionProxy>.Assembly.Location
    
    let obj = appDomain.CreateInstanceFromAndUnwrap(dll, typeof<ReflectionProxy>.FullName) 

    let handler = ResolveEventHandler (fun _ _ -> Assembly.LoadFrom dll)
    do AppDomain.CurrentDomain.add_AssemblyResolve handler

    let rp = obj :?> ReflectionProxy
    let ret = rp.GetDefinedTypes(assemblies)

    do AppDomain.CurrentDomain.remove_AssemblyResolve handler
    do AppDomain.Unload(appDomain)

    ret

[<TypeProvider>]
type ClientProvider (config: TypeProviderConfig) as this =
    inherit TypeProviderForNamespaces ()

    let ns = "SignalRProvider.Hubs"
    let asm = Assembly.GetExecutingAssembly()

    let (typeInfo, clientTypeInfo) = config.ReferencedAssemblies |> List.ofSeq |> getTypes

    let typeNs = "SignalRProvider.Types"
    let types = new System.Collections.Generic.List<ProvidedTypeDefinition>()

    
    let rec getTy (ty: StructuredType) = 
        match ty with
        | Simple(t) -> Type.GetType(t)
        | Complex(typeName, l) ->
            let newTypeName = typeName.Replace('.', '!') // collapse namespaces but keep the full name
            let typeDef = ProvidedTypeDefinition(asm, typeNs, newTypeName, Some typeof<obj>) 

            let setMi = typeof<JsonObject>.GetMethod("Set")

            typeDef.AddMembers(l |> List.map (fun (pName, pTy) -> 
                let pType = getTy pTy
                let set = setMi.MakeGenericMethod(pType)
                let p = ProvidedProperty(pName, 
                                        pType,
                                        SetterCode = (fun [ob; newVal] ->
                                            Expr.Call(set, [ob; Expr.Value(pName); newVal])),
                                        GetterCode = (fun [ob] -> 
                                            <@@ () @@>))
                                            // <@@ JsonObject.Get (%%ob: obj) pName @@>))
                p))
            typeDef.AddMember <| ProvidedConstructor([], InvokeCode =  (fun args -> <@@ JsonObject.Create() @@>))

            types.Add(typeDef)
            upcast typeDef


    let makeMethodType hubName (name, (args: (string * StructuredType) list, ret: StructuredType)) =
        

        let args = args |> List.map (fun (name, ty) -> ProvidedParameter(name, getTy ty))
        let returnType = 
            match ret with
            | Simple(t) when t = typeof<unit>.FullName || t = typeof<System.Void>.FullName -> typeof<unit>
            | _ -> getTy(ret)

        let deferType = typedefof<JQueryDeferred<_>>.MakeGenericType(returnType)

        let objDeferType = typeof<JQueryDeferred<obj>>

        let meth = ProvidedMethod(name, args |> List.ofSeq, objDeferType)

        let castParam (e: Expr) = Expr.Coerce(e, typeof<obj> )

        //let unbox = match <@ 1 :> obj :?> int @> with Call(e, mi, es) -> mi

        meth.InvokeCode <- (fun args -> 
            let argsArray = Expr.NewArray(typeof<obj>, args |> Seq.skip 1 |> Seq.map castParam |> List.ofSeq)

            let objExpr = <@@ let conn = ( %%args.[0] : obj) :?> HubConnection
                              conn.createHubProxy(hubName) @@>

            let invokeExpr = <@@ (%%objExpr : HubProxy).invokeOverload2(name, (%%argsArray: obj array)) @@> 

            invokeExpr)
            
        meth

    let makeHubType (name, methodTypeInfo) =
        let methodDefinedTypes = methodTypeInfo |> Seq.map (makeMethodType name)
        
        let ty = ProvidedTypeDefinition(asm, ns, name, Some typeof<obj>)
        let ctor = ProvidedConstructor(parameters = [ ProvidedParameter("conn", typeof<HubConnection>) ], 
                    InvokeCode = (fun args -> <@@ (%%(args.[0]) : HubConnection) :> obj @@>))
        ty.AddMember ctor
        Seq.iter ty.AddMember methodDefinedTypes 
        ty

        
    let makeClientHubType (hubName, ((name: string, methodTypeInfo): HubType)) =
        let name = if name.StartsWith("I") then name.Substring(1) else name
        let setMi = typeof<JsonObject>.GetMethod("Set")
        let set = setMi.MakeGenericMethod(typeof<obj>)
        let get = typeof<JsonObject>.GetMethod("Get")



        let prop (n: string) (args: (string * StructuredType) list, ret: StructuredType) = 
            let argTys = args |> List.map (fun (name, ty) -> getTy ty)
            let returnType = 
                match ret with
                | Simple(t) when t = typeof<unit>.FullName || t = typeof<System.Void>.FullName -> typeof<unit>
                | _ -> getTy(ret)
            let nameCamelised = n.Substring(0, 1) .ToLower() + n.Substring(1)
            let types = argTys @ [returnType] |> List.toArray
            let tyDef = match types.Length with
                        | n when n < 10 -> Type.GetType("System.Func`" + n.ToString())
                        | _ -> failwith <| "Function with too many params: " + n
            let ty = tyDef.MakeGenericType(types)
            ProvidedProperty(n, ty,
                SetterCode = (fun [ob; newVal] ->
                                                Expr.Call(set, [ob; Expr.Value(nameCamelised); newVal])),
                GetterCode = (fun [ob] -> Expr.Call(get, [ob; Expr.Value(nameCamelised)]))) 
        let methodDefinedTypes =
            methodTypeInfo 
            |> List.map (fun (n, x) -> prop n x)
        let ty = ProvidedTypeDefinition(asm, ns, name, Some typeof<obj>)
        ty.AddMember(ProvidedConstructor([], InvokeCode = (fun _ -> <@@ JsonObject.Create() |> box @@>)))
        Seq.iter ty.AddMember methodDefinedTypes 

        let invoke [obj; arg] =
            let objExpr = <@@ let conn = (%%arg: HubConnection)
                              conn.createHubProxy(hubName) @@>
            Expr.Call(typeof<HubUtil>.GetMethod("RegisterClientProxy"), [objExpr; obj])
        ty.AddMember(ProvidedMethod("Register", [ProvidedParameter("conn", typeof<HubConnection>)], typeof<Void>, InvokeCode = invoke))

        ty

    let definedHubTypes = typeInfo |> List.map makeHubType
    let definedClientHubTypes =
        List.zip (typeInfo |> List.map (fun (n,_) -> n)) clientTypeInfo
        |> List.map makeClientHubType

    do
        this.RegisterRuntimeAssemblyLocationAsProbingFolder(config)
        this.AddNamespace(ns, definedHubTypes)
        this.AddNamespace(ns, definedClientHubTypes)
        this.AddNamespace(typeNs, types |> List.ofSeq)


[<assembly:TypeProviderAssembly>]
do ()