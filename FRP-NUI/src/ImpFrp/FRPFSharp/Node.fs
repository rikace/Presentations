namespace FRPFSharp

open System.Collections.Generic

type Node(rank: int64) =
    let refEqual a b = System.Object.ReferenceEquals(a, b)

    let rank = ref rank
    let listeners = HashSet<Node>()

    static member Null = Node(System.Int64.MaxValue)

    member private this.EnsureBiggerThan(limit: int64, visited: HashSet<_>) =
        if !rank > limit || visited.Contains(this) then
            false
        else
            visited.Add(this) |> ignore
            rank := limit + 1L
            listeners
            |> Seq.iter (fun listener -> listener.EnsureBiggerThan(!rank, visited) |> ignore)

            true
    member this.LinkTo(target: Node) =
        if refEqual target Node.Null then
            false
        else
            let changed = target.EnsureBiggerThan(!rank, HashSet<_>())
            listeners.Add(target) |> ignore
            changed

    member this.UnlinkTo(target: Node) =
        listeners.Remove(target) |> ignore

    member this.Rank = !rank

    interface System.IComparable<Node> with
        member this.CompareTo(other: Node) =
            if !rank < other.Rank then -1
            elif !rank > other.Rank then 1
            else 0
