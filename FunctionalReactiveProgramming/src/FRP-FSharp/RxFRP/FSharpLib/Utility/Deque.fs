namespace Utility

module Deque =

    type Deque<'a>() =    
        let mutable buffer = Array.zeroCreate 16 : 'a[]    
        let mutable mask = 16 - 1    
        let mutable first = 0    
        let mutable next = 0    
        let mutable count  = 0    
        let mutable firstIndex = 0L      

        member inline private t.IncFirst() = first <- (first + 1) &&& mask    
        member inline private t.IncNext()  = next  <- (next  + 1) &&& mask    
        member inline private t.DecFirst() = first <- (first + mask) &&& mask   
        member inline private t.DecNext()  = next  <- (next +  mask) &&& mask      

        member t.Count = count    
        member t.FirstIndex = firstIndex  

        member private t.Grow() =        
            let n = buffer.Length        
            let a = Array.zeroCreate (2*n)       
            System.Array.Copy(buffer, first, a, 0, n - first)       
            System.Array.Copy(buffer, 0, a, n - first, first)        
            buffer <- a        
            first <- 0        
            next <- n        
            mask <- 2*n - 1

        member t.PushFront(e) =        
            t.DecFirst()        
            buffer.[first] <- e        
            let c = count + 1        
            count <- c        
            if c = buffer.Length then t.Grow()        
            firstIndex <- firstIndex - 1L        
            firstIndex

        member t.PushBack(e) =        
            buffer.[next] <- e        
            t.IncNext()        
            let c = count + 1        
            count <- c        
            if c = buffer.Length then t.Grow()        
            firstIndex + int64 (c - 1)

        member t.PopFront()  =        
            if first <> next then            
                let e = buffer.[first]            
                t.IncFirst()            
                count <- count - 1            
                firstIndex <- firstIndex + 1L            
                e        
            else raise (System.InvalidOperationException("The deque is empty.")) 

        member t.PopBack() =        
                                    if first <> next then            
                                        t.DecNext()            
                                        count <- count - 1            
                                        buffer.[next]        
                                    else raise (System.InvalidOperationException("The deque is empty."))

        member t.PeekFront()  =        
            if first <> next then            
                buffer.[first]        
            else raise (System.InvalidOperationException("The deque is empty."))

        member t.PeekBack()  =        
            if first <> next then            
                buffer.[(next + mask) &&& mask]        
            else raise (System.InvalidOperationException("The deque is empty."))

        member t.Item
            with get index   =  let off = index - firstIndex                           
                                if off >= 0L && off < int64 count then                               
                                    buffer.[(first + int32 off) &&& mask]                           
                                else raise (System.ArgumentOutOfRangeException("index"))        
            and  set index v =  let off = index - firstIndex                           
                                if off >= 0L && off < int64 count then                               
                                    buffer.[(first + int32 off) &&& mask] <- v                           
                                else raise (System.ArgumentOutOfRangeException("index"))