namespace Easj360FSharp

open System
open System.Threading 

module PowerEvent =

    [<Sealed>]
    type PowerEvent<'del, 'args
         when 'del :  not struct                  
         and 'del :  delegate<'args, unit>  
         and 'del :> Delegate       
         and 'del :  null>() =      
     
        [<DefaultValue>]  
        val mutable private target : 'del   

        static let invoker : Action<_,_,_> =     
            downcast Delegate.CreateDelegate(         
                typeof<Action<'del, obj, 'args>>,         
                typeof<'del>.GetMethod "Invoke")  
        
        member self.Trigger (sender: obj, args: 'args) =     
            match self.target with
            null    -> ()   
            | handler -> invoker.Invoke (handler, sender, args)   
    
        member self.TriggerAsync (sender: obj, args: 'args) =     
            match self.target with     
            null    -> ()   
            | handler ->         async { invoker.Invoke (handler, sender, args) }         
                                |> Async.Start   
                        
        member self.TriggerParallel (sender: obj, args: 'args) =     
            match self.target with     
            null    -> ()   
            | handler ->         handler.GetInvocationList ()      
                                 |> Array.map (fun h -> async {         
                                            invoker.Invoke (downcast h, sender, args) })      
                                 |> Async.Parallel      
                                 |> Async.Ignore      
                                 |> Async.Start   

        interface IDelegateEvent<'del> with      
            member self.AddHandler handler =       
                self.target <- downcast            
                    Delegate.Combine (self.target, handler)      
    
            member self.RemoveHandler handler =       
                self.target <- downcast            
                   Delegate.Remove (self.target, handler)   

        member self.Publish = self :> IDelegateEvent<'del>   
    
        member self.PublishSync =   
            { new IDelegateEvent<'del> with      
                member __.AddHandler handler =       
                    lock self (fun() ->            
                                self.target <- downcast                 
                                    Delegate.Combine (self.target, handler))      
    
                member __.RemoveHandler handler =       
                    lock self (fun() ->            
                        self.target <- downcast                 
                            Delegate.Remove (self.target, handler)) }   
                

        member self.PublishLockFree =   
            { new IDelegateEvent<'del> with      
                    member __.AddHandler handler =       
                        let rec loop o =         
                            let c = downcast Delegate.Combine (o, handler)         
                            let r = Interlocked.CompareExchange(&self.target,c,o)         
                            if obj.ReferenceEquals (r, o) = false then loop r       
                        loop self.target      
                    
                    member __.RemoveHandler handler =       
                        let rec loop o =         
                            let c = downcast Delegate.Remove (o, handler)         
                            let r = Interlocked.CompareExchange(&self.target,c,o)         
                            if obj.ReferenceEquals (r, o) = false then loop r       
                        loop self.target }