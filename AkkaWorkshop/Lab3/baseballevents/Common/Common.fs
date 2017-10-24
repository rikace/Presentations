module CommonTypes

open System.Runtime.Serialization
open Microsoft.FSharp.Reflection
open System.IO
open System.Reflection
open System.Runtime.Serialization
open System.Runtime.Serialization.Formatters.Binary


[<DataContract>]
type Play =
    {
        [<field:DataMember>]
        id:System.Guid
        [<field:DataMember>]
        gameId:string
        [<field:DataMember>]
        success:int        
        [<field:DataMember>]
        homeTeam:string
        [<field:DataMember>]
        visitingTeam:string
        [<field:DataMember>]
        sequence:string
        [<field:DataMember>]
        inning:int
        [<field:DataMember>]
        balls:int
        [<field:DataMember>]
        strikes:int
        [<field:DataMember>]
        outs:int
        [<field:DataMember>]
        homeScore:int
        [<field:DataMember>]
        visitorScore:int
        [<field:DataMember>]
        rbiOnPlay:int
        [<field:DataMember>]
        hitValue:int
        [<field:DataMember>]
        batter:string
        [<field:DataMember>]
        pitcher:string
        [<field:DataMember>]
        isBatterEvent:bool
        [<field:DataMember>]
        isAtBat:bool
        [<field:DataMember>]
        isHomeAtBat:bool
        [<field:DataMember>]
        isEndGame:bool
        [<field:DataMember>]
        isSacFly:bool
    }

[<DataContract>]
type HandleNewGameEvent =
    {
        [<field:DataMember>]
        id:System.Guid
        [<field:DataMember>]
        gameId:string
        [<field:DataMember>]
        success:int        
        [<field:DataMember>]
        homeTeam:string
        [<field:DataMember>]
        visitingTeam:string
        [<field:DataMember>]
        sequence:string
        [<field:DataMember>]
        inning:int
        [<field:DataMember>]
        balls:int
        [<field:DataMember>]
        strikes:int
        [<field:DataMember>]
        outs:int
        [<field:DataMember>]
        homeScore:int
        [<field:DataMember>]
        visitorScore:int
        [<field:DataMember>]
        rbiOnPlay:int
        [<field:DataMember>]
        hitValue:int
        [<field:DataMember>]
        batter:string
        [<field:DataMember>]
        pitcher:string
        [<field:DataMember>]
        isBatterEvent:bool
        [<field:DataMember>]
        isAtBat:bool
        [<field:DataMember>]
        isHomeAtBat:bool
        [<field:DataMember>]
        isEndGame:bool
        [<field:DataMember>]
        isSacFly:bool
    }

[<DataContract>]
type TeamRunScored =
    {
        [<field:DataMember>]    
        teamId:string
        [<field:DataMember>]    
        runs:int
    }

[<DataContract>]        
type HitterWasAtBat =
    {
        [<field:DataMember>]    
        id:string
        [<field:DataMember>]    
        name:string
        [<field:DataMember>]    
        pitcherId:string
        [<field:DataMember>]    
        hitValue:int
        [<field:DataMember>]    
        rbiOnPlay:int
        [<field:DataMember>]    
        balls:int
        [<field:DataMember>]    
        strikes:int
        [<field:DataMember>]    
        outs:int
        [<field:DataMember>]    
        inning:int
        [<field:DataMember>]    
        isAtBat:bool
        [<field:DataMember>]    
        playType:int
        [<field:DataMember>]    
        isSacrificeFly:bool
        [<field:DataMember>]    
        team:string
    }

[<DataContract>]        
type Batter =
    {
        [<field:DataMember>]    
        atBats:int
        [<field:DataMember>]    
        average:float
        [<field:DataMember>]    
        hits:int
        [<field:DataMember>]    
        batterId:string
        [<field:DataMember>]    
        playerId:string
        [<field:DataMember>]    
        name:string
        [<field:DataMember>]    
        onBase:float
        [<field:DataMember>]    
        sacrificeFlies:int
        [<field:DataMember>]    
        slugging:float
        [<field:DataMember>]    
        walks:int
        [<field:DataMember>]    
        hitByPitch:int
        [<field:DataMember>]    
        totalBases:int
    }

[<DataContract>]        
type PitcherFacedBatter =
    {
        [<field:DataMember>]  
        pictherId:string
        [<field:DataMember>]  
        name:string
        [<field:DataMember>]  
        hitValue:int
        [<field:DataMember>]  
        isAtBat:bool
        [<field:DataMember>]  
        playType:int
        [<field:DataMember>]  
        isSacrificeFly:bool
    }

[<DataContract>]        
type Pitcher =
    {
        [<field:DataMember>]  
        pictherId:string
        [<field:DataMember>]  
        name:string        
        [<field:DataMember>]  
        hits:int
        [<field:DataMember>]  
        walks:int
        [<field:DataMember>]  
        atBats:int
        [<field:DataMember>]  
        totalBases:int
        [<field:DataMember>]  
        hitByPitch:int
        [<field:DataMember>]  
        oppAvg:float
        [<field:DataMember>]  
        oppObp:float
        [<field:DataMember>]  
        oppSlugging:int
        [<field:DataMember>]  
        sacrificeFlies:int
    }        

type PlayType =
| Unknown = 0
| None = 1
| Out = 2
| Strikeout = 3
| StolenBase = 4
| DefensiveIndifference = 5
| CaughtStealing = 6
| PickoffError = 7
| Pickoff = 8
| WildPitch = 9
| PassedBall = 10
| Balk = 11
| OtherAdvance = 12
| FoulError = 13
| Walk = 14
| IntentionalWalk = 15
| HitByPitch = 16
| Interference = 17
| Error = 18
| FieldersChoice = 19
| Single = 20
| Double = 21
| Triple = 22
| HomeRun = 23
| MissingRlay = 24
    