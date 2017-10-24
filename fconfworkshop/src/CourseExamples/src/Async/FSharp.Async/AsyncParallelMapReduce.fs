module AsyncObserver

open System
open System.Net
open Microsoft.FSharp.Control.WebExtensions
open FSharp.Collections.ParallelSeq


let noiseWords = [|"a"; "about"; "above"; "all"; "along"; "also"; "although"; "am"; "an"; "any"; "are"; "aren't"; "as"; "at";
            "be"; "because"; "been"; "but"; "by"; "can"; "cannot"; "could"; "couldn't";
            "did"; "didn't"; "do"; "does"; "doesn't"; "e.g."; "either"; "etc"; "etc."; "even"; "ever";
            "for"; "from"; "further"; "get"; "gets"; "got"; "had"; "hardly"; "has"; "hasn't"; "having"; "he";
            "hence"; "her"; "here"; "hereby"; "herein"; "hereof"; "hereon"; "hereto"; "herewith"; "him";
            "his"; "how"; "however"; "I"; "i.e."; "if"; "into"; "it"; "it's"; "its"; "me"; "more"; "most"; "mr"; "my";
            "near"; "nor"; "now"; "of"; "onto"; "other"; "our"; "out"; "over"; "really"; "said"; "same"; "she"; "should";
            "shouldn't"; "since"; "so"; "some"; "such"; "than"; "that"; "the"; "their"; "them"; "then"; "there"; "thereby";
            "therefore"; "therefrom"; "therein"; "thereof"; "thereon"; "thereto"; "therewith"; "these"; "they"; "this";
            "those"; "through"; "thus"; "to"; "too"; "under"; "until"; "unto"; "upon"; "us"; "very"; "viz"; "was"; "wasn't";
            "we"; "were"; "what"; "when"; "where"; "whereby"; "wherein"; "whether"; "which"; "while"; "who"; "whom"; "whose";
            "why"; "with"; "without"; "would"; "you"; "your" ; "have"; "thou"; "will"; "shall"|]


let links =
    [|  "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/allswellthatendswell.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/amsnd.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/antandcleo.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/asyoulikeit.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/comedyoferrors.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/cymbeline.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/hamlet.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryiv1.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryiv2.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryv.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryvi1.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryvi2.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryvi3.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/henryviii.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/juliuscaesar.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/kingjohn.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/kinglear.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/loveslobourslost.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/maan.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/macbeth.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/measureformeasure.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/merchantofvenice.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/othello.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/richardii.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/richardiii.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/romeoandjuliet.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/tamingoftheshrew.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/tempest.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/themwofw.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/thetgofv.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/timon.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/titus.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/troilusandcressida.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/twelfthnight.txt";
        "http://www.cs.ukzn.ac.za/~hughm/ap/data/shakespeare/winterstale.txt"|]

let Download(url : Uri) =
    async {
        let client = new WebClient()
        let! html = client.AsyncDownloadString(url)
        return html
    }

let (<||>) first second = async { let! results = Async.Parallel([|first; second|])
                                                 in return (results.[0], results.[1]) }

let add x = let res = x in res + x


(*[/omit]*)

// A balanced binary-tree representation of the parallel execution of mapReduce
//           r
//          / \
//         /   \
//        /     \
//       /       \
//      /         \
//     r           r
//    /\          /\
//   /  \        /  \
//  /    \      /    \
// f a1  f a2  f a3  f a4


//(|f ,⊗|)
let mapReduce (mapF : 'T -> Async<'R>) (reduceF : 'R -> 'R -> Async<'R>) (input : 'T []) : Async<'R> =
    let rec mapReduce' s e =
        async {
            if s + 1 >= e then return! mapF input.[s]
            else
                let m = (s + e) / 2
                let! (left, right) =  mapReduce' s m <||> mapReduce' m e
                return! reduceF left right
        }
    mapReduce' 0 input.Length


// Example: the classic map/reduce word-count

let mapF uri =
    async {
        let! text = Download(new Uri(uri))
        let words = text.Split([|' '; '.'; ','|], StringSplitOptions.RemoveEmptyEntries)
        return
            words
            |> PSeq.map (fun word -> word.ToUpper())
            |> PSeq.filter (fun word -> not (noiseWords |> Seq.exists (fun noiseWord -> noiseWord.ToUpper() = word)) && Seq.length word > 3)
            |> PSeq.groupBy id
            |> PSeq.map (fun (key, values) -> (key, values |> PSeq.length)) |> PSeq.toList
    }

let reduceF (left : (string * int) list) (right : (string * int) list) =
    async {
        return
            left @ right
            |> PSeq.groupBy fst
            |> PSeq.map (fun (key, values) -> (key, values |> PSeq.sumBy snd))
            |> PSeq.toList
    }




let mr = mapReduce mapF reduceF links
let res = mr |> Async.RunSynchronously
                |> List.sortBy (fun (_, count) -> -count)

printfn "%A" res

