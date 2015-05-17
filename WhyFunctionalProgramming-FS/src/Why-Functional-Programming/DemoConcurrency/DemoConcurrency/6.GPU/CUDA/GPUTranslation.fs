module GPUTranslation

open System
open System.Reflection
open Microsoft.FSharp.Quotations
open Microsoft.FSharp.Quotations.Patterns
open Microsoft.FSharp.Quotations.DerivedPatterns
open GPUTranslator
open Microsoft.FSharp.Quotations.ExprShape
open Microsoft.FSharp.Reflection
open ItemType
open CudaDataStructure
 
let resultVariableName = "resultOut"
let code = new Code()

let isGPUFunction (exp:Expr) = 
    let isfun = FSharpType.IsFunction(exp.Type)
    let isgpu = match exp with
                  | DerivedPatterns.Lambdas(c, Patterns.Call(a, mi, b)) ->
                      let attrs = mi.GetCustomAttributes(typeof<GPUAttribute>, false)
                      attrs.Length > 0
                  | _ -> 
                      failwith "Argument must be of the form <@ foo @>!" 
    isfun && isgpu

let rec getFunctionBody (exp:Expr) = 
    match exp with
    | DerivedPatterns.Lambdas(c, callPattern) ->
        match callPattern with
        | Patterns.Call (e, mi, exprList) ->
            match mi with
                | DerivedPatterns.MethodWithReflectedDefinition n -> callPattern //getFunctionBody n
                | _ -> callPattern       
        | Patterns.Sequential _ -> callPattern
        | _ -> callPattern
    | Patterns.Sequential _ -> exp
    | _ -> failwith "Argument must be of the form <@ foo @>!" 

let getFunctionParameterAndReturn (exp:Expr) =
    match exp with
        | DerivedPatterns.Lambdas (c, Patterns.Call(a, mi, b)) ->
            Some(b, mi.ReturnType)
        | _ -> None

let getFunctionName (exp:Expr) = 
    match exp with
        | DerivedPatterns.Lambdas(c, Patterns.Call(a, mi, b)) ->
            mi.Name
        | _ -> 
            failwith "Argument must be of the form <@ foo @>!" 

let getFunctionTypes (exp:Expr) = 
    match getFunctionParameterAndReturn(exp) with
        | Some(exprList ,t) -> 
            let out = exprList |> List.map (fun n -> (n, n.Type))
            Some(out, t)
        | None -> None

let getFunctionReturnType (exp:Expr) = 
    match getFunctionTypes(exp) with
    | Some(_, t) -> t
    | _ -> failwith "cannot find return type"

let isTypeGPUOK t = 
    if t = typeof<int> then true
    elif t = typeof<float32> then true
    elif t = typeof<float> then true
    else false

let isValueGPUOK (v:obj) = 
    match v with 
    | :? int
    | :? float32 -> true
    | :? float -> true
    | _ -> false

let rec getCallReturnType (exp:Expr) = 
    match exp with
    | Patterns.Call (_, mi, _) -> mi.ReturnType
    | Patterns.Var n -> n.Type
    | Patterns.Let (var, e0, e1) -> getCallReturnType e1
    | Patterns.Sequential (e0, e1) -> getCallReturnType e1
    | Patterns.Value(v) -> snd v
    | Patterns.WhileLoop(e0, e1) -> typeof<System.Void>
    | Patterns.ForIntegerRangeLoop(var, e0, e1, e2) -> getCallReturnType e2
    | Patterns.IfThenElse(e0, e1, e2) -> typeof<System.Void>
    | _ -> failwith "not supported expr type"


let getFunctionSignature (exp:Expr) = 
    let template = @"extern ""C"" __global__ void {0} ({1}) "
    let functionName = getFunctionName(exp)
    let parameters = getFunctionParameterAndReturn(exp)
    match parameters with 
    | Some(exprList, _) -> 
        let parameterNames = 
            exprList 
            |> Seq.map (fun n -> 
                        match n with 
                        | _ when n.Type.IsAssignableFrom(typeof<CUDAPointer2<float>>) -> sprintf "double* %s" (n.ToString())
                        | _ when n.Type.IsAssignableFrom(typeof<CUDAPointer2<float32>>) -> sprintf "float* %s" (n.ToString())
                        | _ -> sprintf "%s %s" (translateFromNETType n.Type (n.ToString())) (n.ToString()) )
        String.Format(template, functionName, String.Join(", ", parameterNames))        
    | None -> failwith "cannot get parameter and return type"

let getFunctionSignatureFromMethodInfo (mi:MethodInfo) = 
    let template = @"extern ""C"" __global__ void {0} ({1}) "
    let functionName = mi.Name
    let parameters = 
        mi.GetParameters() 
        |> Array.map (fun n -> 
                        match n with 
                        | _ when n.ParameterType.IsAssignableFrom(typeof<CUDAPointer2<float>>) -> sprintf "double* %s" n.Name
                        | _ when n.ParameterType.IsAssignableFrom(typeof<CUDAPointer2<float32>>) -> sprintf "float* %s" n.Name
                        | _ -> sprintf "%s %s" (n.ParameterType.Name) n.Name )
    String.Format(template, functionName, String.Join(", ", parameters))

let accessExpr exp = 

    let addSemiCol (str:string) = 
            if str.EndsWith(";") || String.IsNullOrEmpty(str) && String.IsNullOrWhiteSpace(str) then str
            else str + ";"

    let rec iterate exp : string=
        
        let print x = 
            let str = sprintf "%A" x
            str

        let matchExp expOption = 
            match expOption with
            | Some(n) -> iterate n
            | None -> String.Empty

        let isCUDAPointerType (exp:Expr Option) = 
            match exp with
            | Some(n) -> n.Type.IsAssignableFrom(typeof<CUDAPointer2<float>>) || n.Type.IsAssignableFrom(typeof<CUDAPointer2<float32>>)
            | _ -> false        
    
        match exp with
        | DerivedPatterns.Applications (e, ell) -> 
            let str0 = iterate e
            let str1 = 
                ell 
                |> Seq.map (fun n -> n |> Seq.map (fun m -> iterate m ))
                |> Seq.map (fun n -> String.Join("\r\n", n |> Seq.toArray))
            str0 + String.Join("\r\n", str1 |> Seq.toArray)
        | DerivedPatterns.AndAlso (e0, e1) -> 
            (iterate e0) + (iterate e1)
        | DerivedPatterns.Bool e ->
            print e
        | DerivedPatterns.Byte e ->
            print e
        | DerivedPatterns.Char e ->
            print e
        | DerivedPatterns.Double e ->
            print e
        | DerivedPatterns.Int16 e->
            print e
        | DerivedPatterns.Int32 e->
            print e
        | DerivedPatterns.Int64 e ->
            print e
        | DerivedPatterns.OrElse (e0, e1)->
            (iterate e0) + (iterate e1)
        | DerivedPatterns.SByte e ->
            print e
        | DerivedPatterns.Single e ->
            print e
        | DerivedPatterns.String e ->
            print e
        | DerivedPatterns.UInt16 e -> 
            print e
        | DerivedPatterns.UInt32 e -> 
            print e
        | DerivedPatterns.UInt64 e -> 
            print e
        | DerivedPatterns.Unit e ->
            String.Empty //"void"
        | Patterns.AddressOf address ->
            iterate address
        | Patterns.AddressSet (exp0, exp1) ->
            (iterate exp0) + (iterate exp1)
        | Patterns.Application (exp0, exp1) ->
            (iterate exp0) + (iterate exp1)
        | Patterns.Call (expOption, mi, expList)  ->
            if isCUDAPointerType expOption && mi.Name="Set" then
                let callObject = matchExp expOption
                let index = iterate expList.[1]
                let postfix = 
                    match mi with
                    | DerivedPatterns.MethodWithReflectedDefinition n -> iterate n
                    | _ -> iterate expList.[0]
                let s = sprintf "%s[%s] = %s;" callObject index postfix
                s                
            else
                let callObject = matchExp expOption
                let returnType = translateFromNETType mi.ReturnType String.Empty
                let postfix = 
                    match mi with
                    | DerivedPatterns.MethodWithReflectedDefinition n -> iterate n
                    | _ -> translateFromNETOperator mi expList
                let s = sprintf "%s%s" callObject postfix
                s
        | Patterns.Coerce (exp, t)-> 
            let from = iterate exp         
            //sprintf "coerce(%s, %s)" from t.Name
            sprintf "%s" from
        | Patterns.DefaultValue exp -> 
            print exp
        | Patterns.FieldGet (expOption, fi) ->
            (matchExp expOption) + (print fi)
        | Patterns.FieldSet (expOption, fi, e) ->
            let callObj = matchExp expOption
            let fi = print fi
            let str = iterate e
            callObj + fi + str
        | Patterns.ForIntegerRangeLoop (v, e0, e1, e2) ->
            let from = iterate e0
            let toV = iterate e1
            let s = String.Format("for (int {0} = {1}; {0}<{2}; {0}++) {{ {3} }}", v, from ,toV, iterate e2)
            s
        | Patterns.IfThenElse (con, exp0, exp1) ->
            let condition = (iterate con)
            let ifClause = addSemiCol(iterate exp0)
            let elseClause = addSemiCol(iterate exp1)            
            sprintf "if (%s) { %s }\r\nelse { %s }" condition ifClause elseClause
        | Patterns.Lambda (var,body) ->
            //let a = print var
            //let b = iterate body
            match exp with 
            | DerivedPatterns.Lambdas (vll, e) ->
                let s = 
                    vll 
                    |> List.map (fun n-> n |> List.map (fun m -> sprintf "%s %s" (translateFromNETType m.Type "") m.Name))
                    |> List.fold (fun acc l -> acc@l) []
                let parameterNames = vll |> List.map (fun n -> sprintf "%s" n.Head.Name)
                let returnType = getCallReturnType e
                let returnTypeID = translateFromNETTypeToFunReturn returnType ""
                let fid = code.FunctionID;
                code.IncreaseFunctionID()
                let functionName = sprintf "ff_%d" fid
                let statement = iterate e
                let functionCode = sprintf "__device__ %s %s(%s) { %s } " returnTypeID functionName (String.Join(", ", s)) (addSemiCol(statement))
                code.Add(functionCode)
                sprintf "%s(%s)" functionName (String.Join(", ", parameterNames))
            | _ -> failwith "not supported lambda format"           
        | Patterns.Let (var, exp0, exp1) ->
            let a = print var
            let b = iterate exp0
            let t = var.Type
            let s = 
                if t.Name = "FSharpFunc`2" then
                    sprintf "__device__ %s; //function pointer" (translateFromNETType t a)
                else
                    String.Empty
            code.Add(s)
            let c = iterate exp1
            let assignment = 
                if t.Name = "FSharpFunc`2" then
                    sprintf "%s;\r\n%s = %s;" (translateFromNETType t a) a b                    
                else
                    sprintf "%s %s;\r\n%s = %s;" (translateFromNETType t a) a a b
            sprintf "%s\r\n%s" assignment c
        | Patterns.LetRecursive (tupList, exp) ->
            let strList = tupList |> Seq.map (fun (var, e) -> (print var) + (iterate e))
            String.Join("\r\n", strList |> Seq.toArray) + (iterate exp)
        | Patterns.NewArray (t, expList) ->
            let str0 = print t
            let str1 = expList |> Seq.map (fun e -> iterate e)
            str0 + String.Join("\r\n", str1)
        | Patterns.NewDelegate (t, varList, exp) ->
            (print t) + (print varList) + (iterate exp)
        | Patterns.NewObject (t, expList) -> 
            let str0 = print t
            let str1 = expList |> Seq.map (fun e -> iterate e)
            str0 + String.Join("\r\n", str1)
        | Patterns.NewRecord (t, expList) -> 
            let str0 = print t
            let str1 = expList |> Seq.map (fun e -> iterate e)
            str0 + String.Join("\r\n", str1)
        | Patterns.NewObject (t, expList) -> 
            let str0 = print t
            let str1 = expList |> Seq.map (fun e -> iterate e)
            str0 + String.Join("\r\n", str1)
        | Patterns.NewRecord (t, expList) -> 
            let str0 = print t
            let str1 = expList |> Seq.map (fun e -> iterate e)
            str0 + String.Join("\r\n", str1)
        | Patterns.NewTuple expList -> 
            let ty = translateFromNETType (expList.[0].Type) String.Empty
            let l = expList |> Seq.map (fun e -> iterate e)
            let l = String.Join(", ", l)
            sprintf "newTuple<%s>(%s)" ty l
        | Patterns.NewUnionCase (t, expList) -> 
            let str0 = print t
            let str1 = expList |> Seq.map (fun e -> iterate e)
            str0 + String.Join("\r\n", str1)
        | Patterns.PropertyGet (expOption, pi, expList) ->
            let callObj = matchExp expOption
            let r = match pi with 
                    | DerivedPatterns.PropertyGetterWithReflectedDefinition e -> 
                        iterate e
                    | _ -> pi.Name
            let l = expList |> List.map (fun n -> iterate n)
            if l.Length > 0 then
                if r = "Item" then
                    sprintf "%s[%s]" callObj (String.Join(", ", l))
                else
                    sprintf "%s.%s[%s]" callObj r (String.Join(", ", l))
            else
                if String.IsNullOrEmpty callObj then
                    sprintf "%s" r
                else
                    sprintf "%s.%s" callObj r
        | Patterns.PropertySet (expOption, pi, expList, e) ->
            let callObj = matchExp expOption
            let r = match pi with 
                    | DerivedPatterns.PropertyGetterWithReflectedDefinition e -> 
                        iterate e
                    | _ -> print pi
            let l = expList |> Seq.map (fun n -> iterate n)
            if r = "Item" then
                callObj + String.Join("\r\n", l) + (iterate e)
            else
                callObj + r + String.Join("\r\n", l) + (iterate e)
        | Patterns.Quote e ->
            iterate e
        | Patterns.Sequential (e0, e1) ->
            let statement0 = addSemiCol(iterate e0)
            let statement1 = addSemiCol(iterate e1)
            sprintf "%s\r\n%s" statement0 statement1
        | Patterns.TryFinally (e0, e1) ->
            (iterate e0) + (iterate e1)
        | Patterns.TryWith (e0, v0, e1, v1, e2) -> 
            (iterate e0) + (print v0) + (iterate e1) + (print v1) + (iterate e2)
        | Patterns.TupleGet (e, i) -> 
            (iterate e) + (print i)
        | Patterns.TypeTest (e, t) ->
            (iterate e) + (print t)
        | Patterns.UnionCaseTest (e, ui) ->
            (iterate e) + (print ui)
        | Patterns.Value (obj, t) ->
            (print obj) + (print t)
        | Patterns.Var v ->
            v.Name
        | Patterns.VarSet (v, e) ->
            let left = (print v)
            let right = (iterate e)
            sprintf "%s = %s" left right
        | Patterns.WhileLoop (e0, e1) ->
            let condition = iterate e0
            let body = iterate e1
            sprintf "while (%s) { \r\n %s \r\n}" condition (addSemiCol(body))
        | _ -> failwith "not supported pattern"
    and translateFromNETOperator (mi:MethodInfo) (exprList:Expr list) = 
        let getList() = exprList |> List.map (fun n -> iterate n) 
        let ty = translateFromNETType (exprList.[0].Type) String.Empty

        let generateFunction (mi:MethodInfo) (mappedMethodName:string) (parameters:Expr list)=                                    
            let result = sprintf "%s(%s)" mappedMethodName (String.Join(", ", getList()))
            result

        match mi.Name with
            | "op_Addition" -> 
                let l = getList()
                sprintf "(%s) + (%s)" l.[0] l.[1]
            | "op_Subtraction" -> 
                let l = getList()
                sprintf "(%s) - (%s)" l.[0] l.[1]
            | "op_Multiply" -> 
                let l = getList()
                sprintf "(%s) * (%s)" l.[0] l.[1]
            | "op_Division" -> 
                let l = getList()
                sprintf "(%s) / (%s)" l.[0] l.[1]
            | "op_LessThan" -> 
                let l = getList()
                sprintf "(%s) < (%s)" l.[0] l.[1]
            | "op_LessThanOrEqual" ->
                let l = getList()
                sprintf "(%s) <= (%s)" l.[0] l.[1]
            | "op_GreaterThan" ->
                let l = getList()
                sprintf "(%s) > (%s)" l.[0] l.[1]
            | "op_GreaterThanOrEqual" ->
                let l = getList()
                sprintf "(%s) >= (%s)" l.[0] l.[1]
            | "op_Range" -> failwith "not support range on GPU"
            | "op_Equality" -> 
                let l = getList()
                sprintf "(%s) == (%s)" l.[0] l.[1]
            | "GetArray" -> 
                let l = getList()
                sprintf "%s[%s]" l.[0] l.[1]
            | "CreateSequence" -> failwith "not support createSeq on GPU"
            | "FailWith" -> failwith "not support exception on GPU"
            | "ToList" -> failwith "not support toList on GPU"
            | "Map" -> failwith "not support map on GPU"
            | "Delay" -> 
                let l = getList()
                String.Join(", ", l)
            | "op_PipeRight" -> 
                let l = getList()
                sprintf "%s ( %s )" l.[1] l.[0]
            | "ToSingle" ->
                let l = getList()
                sprintf "(float) (%s)" l.[0]
            | _ -> 
                let l = getList()
                sprintf ".%s(%s)" (mi.Name) (String.Join(", ", l))

    let s = iterate exp    
    addSemiCol(s)

let rec getSequentialStatements (expr:Expr) = 
    match expr with
        | Patterns.Sequential (e0, e1) -> 
            (getSequentialStatements e0) @ [e1]
        | _ -> [expr]

let getCUDACode(f) = 
    let s = getFunctionSignature f
    let body = f |> getFunctionBody 
    let functionStr = sprintf "%s {\r\n%s\r\n}\r\n" s (accessExpr body)
    sprintf "%s" functionStr

let getCommonCode() = code.ToCode()

let getGPUFunctions() = 
    let currentAssembly = Assembly.GetExecutingAssembly()
    let gpuMethods = 
        currentAssembly.GetTypes()
        |> List.ofArray
        |> List.collect (fun t -> t.GetMethods() |> List.ofArray)
        |> List.filter (fun mi -> mi.GetCustomAttributes(typeof<GPUAttribute>, true).Length > 0)        
    //gpuMethods

    gpuMethods 
    |> List.map (fun mi -> (mi, (Quotations.Expr.TryGetReflectedDefinition mi)))    
    |> List.map (fun (mi, Some(expr)) -> (mi, expr))
    |> List.map (fun (mi, expr) -> sprintf "%s {\r\n %s \r\n}" (getFunctionSignatureFromMethodInfo mi) (accessExpr expr))

let tempFile = "temp1.cu"

let GenerateCodeToFile() =     
    let gpuCode = getGPUFunctions()    
    let commonCode = getCommonCode()
    let allCode = String.Join("\r\n", commonCode :: gpuCode)
    System.IO.File.Delete(tempFile)
    System.IO.File.WriteAllText(tempFile, allCode);
