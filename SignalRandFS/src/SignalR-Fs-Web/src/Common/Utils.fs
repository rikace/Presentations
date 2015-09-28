namespace Common

open System
open System.Threading.Tasks


[<AutoOpen>]
module Utils =

    type IId = abstract Id:Guid

    // Infix operator to remove a specific item from a list
    let (--) needle = List.filter ((<>) needle)
    let (++) = List.append

    let replace (needle:string) (replacement:string) (str:string) = str.Replace(needle, replacement)
    let lcasefirst (str:string) = str.Substring(0, 1).ToLower() + str.Substring(1)

    let pascalCase = 
        let replaceDashWithSpace = replace "-" " "
        let replaceSpaceWithNothing = replace " " ""
        
        replaceDashWithSpace >> 
        System.Globalization.CultureInfo.CurrentCulture.TextInfo.ToTitleCase >> 
        replaceSpaceWithNothing >> 
        lcasefirst
        
        
    let memoize f =
        let dict = new System.Collections.Generic.Dictionary<_,_>()
        fun n ->
            match dict.TryGetValue(n) with
            | (true, v) -> v
            | _ ->
                let temp = f(n)
                dict.Add(n, temp)
                temp

    let rec retry count interval (isRetryException:System.Exception->bool) (work:Async<'T>) = 
        async { 
            try 
                let! result = work
                return Choice1Of2 result
            with e ->
                if isRetryException e && count > 0 then
                    do! Async.Sleep interval  
                    return! retry (count - 1) interval isRetryException work
                else 
                    return Choice2Of2 e
        }

