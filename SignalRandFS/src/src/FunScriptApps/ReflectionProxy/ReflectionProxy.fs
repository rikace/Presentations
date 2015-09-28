namespace ReflectionProxy

open System.Reflection
open System
open System.IO

open Microsoft.FSharp.Reflection

type StructuredType = Simple of string | Complex of string * (String * StructuredType) list
type MethodType = (string * StructuredType) list * StructuredType
type HubType = string * (string * MethodType) list

type ReflectionProxy() =
    inherit MarshalByRefObject()

    let mutable assembliesVisited = Set.empty

    let loadAssemblyBytes ass = 
        let resolve (o: obj) (args: ResolveEventArgs) = 
            let name = AssemblyName(args.Name).Name
            try
                Assembly.Load args.Name
            with
                | _ -> Assembly.LoadFrom <| Path.Combine(Path.GetDirectoryName ass, name + ".dll") 
            

        AppDomain.CurrentDomain.add_AssemblyResolve (ResolveEventHandler resolve)
        let res = Assembly.LoadFrom ass
        AppDomain.CurrentDomain.remove_AssemblyResolve (ResolveEventHandler resolve)
        res

    let getMethodType (hubName : string) (mi: MethodInfo) =
        let name = mi.Name
        let parms = mi.GetParameters() |> Seq.map (fun p -> (p.Name,  p.ParameterType (* typeof<obj> *)))
        let returnType = if mi.ReturnType.Equals(typeof<System.Void>) then typeof<unit> else mi.ReturnType
        (name, parms, returnType)

    let getRecordFieldType (field: PropertyInfo) =
        let ty = field.PropertyType 
        if FSharpType.IsFunction ty then
            let (argty, retty) = FSharpType.GetFunctionElements ty
            Some(field.Name, [("arg", argty)] |> Seq.ofList, retty)
        else
            None
    
    // whatever version
    let nameHubNameAttr = "Microsoft.AspNet.SignalR.Hubs.HubNameAttribute"
    let nameIHub = "Microsoft.AspNet.SignalR.Hubs.IHub"
    let nameHub = "Microsoft.AspNet.SignalR.Hubs.HHub"

    let hubAttrs (t: Type) = 
        t.GetCustomAttributes()
        |> Seq.filter (fun attr -> attr.GetType().FullName = nameHubNameAttr)

    let hubName (hubType : Type) = 
        match hubAttrs hubType |> List.ofSeq with
        | attr :: t -> attr.GetType().GetProperty("HubName").GetValue(attr) :?> string
        | [] -> hubType.Name

    let rec encodeType (t: Type) = 
        if t.IsPrimitive || List.exists (fun x -> x = t) [ typeof<unit>; typeof<DateTime>; typeof<string> ] then Simple t.FullName
        else 
            let props = t.GetProperties(BindingFlags.Instance ||| BindingFlags.Public) |> List.ofArray |> List.map getPropertyValue
            let fields = t.GetFields(BindingFlags.Instance ||| BindingFlags.Public) |> List.ofArray |> List.map getFieldValue            
            Complex(t.FullName, (props @ fields))
    and getPropertyValue (pi: PropertyInfo) = (pi.Name, encodeType pi.PropertyType)
    and getFieldValue (pi: FieldInfo) = (pi.Name, encodeType pi.FieldType)


    let makeHubType hubType : HubType =
        let name = hubName hubType

        // exclusion taken from signalr defn
        let excludeTypes = [ nameHub; typeof<obj>.FullName ]
        let excludeInterfaces = [ nameIHub; typeof<IDisposable>.FullName]

        let findty tname = hubType.GetTypeInfo().ImplementedInterfaces |> Seq.tryFind (fun i -> i.FullName = tname)

        let exclude (m: MethodInfo) =
            //m.IsSpecialName             ||
             excludeTypes |> List.exists (fun x -> m.GetBaseDefinition().DeclaringType.FullName = x) 
            || excludeInterfaces 
                |> Seq.collect (fun ity ->
                    match findty ity with 
                    | Some(t) -> hubType.GetInterfaceMap(t).TargetMethods
                    | None -> [||]) 
                |> Seq.exists (fun x -> x = m) 

        let methTypes = 
            if FSharpType.IsRecord hubType then
                let fields = FSharpType.GetRecordFields(hubType)
                fields |> Seq.choose getRecordFieldType |> List.ofSeq
            else
                hubType.GetMethods(BindingFlags.Public ||| BindingFlags.Instance)
                |> Seq.where (exclude >> not)
                |> Seq.map (getMethodType name) 
                |> List.ofSeq

        let methTypeNames : (string * MethodType) list = 
            methTypes 
            |> List.map (fun (name,args,retty) -> (name, (args |> List.ofSeq |> List.map (fun (n,ty) -> (n, encodeType ty)), encodeType retty)))

        (name, methTypeNames)

    let findHubs types = 
        let hasHubAttribute = hubAttrs >> Seq.isEmpty >> not
        List.filter hasHubAttribute types

    let findClientHubDefs (types : Type list) hubName =
        types |> List.filter  (fun t -> String.Equals(t.Name, "I" + hubName + "Client", StringComparison.OrdinalIgnoreCase))

    member this.GetDefinedTypes(assemblies : string seq)  = 
        let asm = assemblies 
        if (not (Seq.isEmpty asm)) then
            let clientAsm = loadAssemblyBytes (Seq.last asm)
            let types = clientAsm.ExportedTypes |> List.ofSeq
            let hubTypes = types |> findHubs |> List.map makeHubType
            let names = List.map (fun (a,b) -> a) hubTypes
            let clientHubDefs = List.collect (findClientHubDefs types) names |> List.map makeHubType
            (hubTypes, clientHubDefs)
        else 
            ([], [])
            