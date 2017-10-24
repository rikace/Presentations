namespace DataModel
    open System.Data.Entity

    type MedalValue() =
        let mutable games = Unchecked.defaultof<string>
        let mutable year = Unchecked.defaultof<int>
        let mutable sport = Unchecked.defaultof<string>
        let mutable discipline = Unchecked.defaultof<string>
        let mutable athlete = Unchecked.defaultof<string>
        let mutable team = Unchecked.defaultof<string>
        let mutable gender = Unchecked.defaultof<string>
        let mutable event = Unchecked.defaultof<string>
        let mutable metal = Unchecked.defaultof<string>
        let mutable gold = Unchecked.defaultof<int>
        let mutable silver = Unchecked.defaultof<int>
        let mutable bronze = Unchecked.defaultof<int>

        member x.Games with get() = games and set(v) = games <- v
        member x.Year with get() = year and set(v) = year <- v
        member x.Sport with get() = sport and set(v) = sport <- v
        member x.Discipline with get() = discipline and set(v) = discipline <- v
        member x.Athlete with get() = athlete and set(v) = athlete <- v
        member x.Team with get() = team and set(v) = team <- v
        member x.Gender with get() = gender and set(v) = gender <- v
        member x.Event with get() = event and set(v) = event <- v
        member x.Metal with get() = metal and set(v) = metal <- v
        member x.Gold with get() = gold and set(v) = gold <- v
        member x.Silver with get() = silver and set(v) = silver <- v
        member x.Bronze with get() = bronze and set(v) = bronze <- v

    type Medal() =
        [<System.ComponentModel.DataAnnotations.Key>]
        member val Id = Unchecked.defaultof<int> with get, set
        member val Games = Unchecked.defaultof<string> with get, set
        member val Year = Unchecked.defaultof<int> with get, set
        member val Sport = Unchecked.defaultof<string> with get, set
        member val Discipline = Unchecked.defaultof<string> with get, set
        member val Athlete = Unchecked.defaultof<string> with get, set
        member val Team = Unchecked.defaultof<string> with get, set
        member val Gender = Unchecked.defaultof<string> with get, set
        member val Event = Unchecked.defaultof<string> with get, set
        member val Metal = Unchecked.defaultof<string> with get, set
        member val Gold = Unchecked.defaultof<int> with get, set
        member val Silver = Unchecked.defaultof<int> with get, set
        member val Bronze = Unchecked.defaultof<int> with get, set

    type OlympicDbContext(connString:string) =
        inherit DbContext(connString)

        [<DefaultValue>]
        val mutable medals : DbSet<Medal>
        member self.Medals with get() = self.medals
                           and set value = self.medals <- value




    [<RequireQualifiedAccess>]
    module TestTable =
        type Plate() =
            let mutable games = Unchecked.defaultof<string>
            let mutable year = Unchecked.defaultof<int>
    //        let mutable sport = Unchecked.defaultof<string>
            let mutable discipline = Unchecked.defaultof<string>
            let mutable athlete = Unchecked.defaultof<string>
            let mutable team = Unchecked.defaultof<string>
            let mutable gender = Unchecked.defaultof<string>
            let mutable event = Unchecked.defaultof<string>
            let mutable metal = Unchecked.defaultof<string>
            let mutable gold = Unchecked.defaultof<int>
            let mutable silver = Unchecked.defaultof<int>
            let mutable bronze = Unchecked.defaultof<int>

            member x.Games with get() = games and set(v) = games <- v
            member x.Year with get() = year and set(v) = year <- v
      //      member x.Sport with get() = sport and set(v) = sport <- v
            member x.Discipline with get() = discipline and set(v) = discipline <- v
            member x.Athlete with get() = athlete and set(v) = athlete <- v
            member x.Team with get() = team and set(v) = team <- v
            member x.Gender with get() = gender and set(v) = gender <- v
            member x.Event with get() = event and set(v) = event <- v
            member x.Metal with get() = metal and set(v) = metal <- v
            member x.Gold with get() = gold and set(v) = gold <- v
            member x.Silver with get() = silver and set(v) = silver <- v
            member x.Bronze with get() = bronze and set(v) = bronze <- v
