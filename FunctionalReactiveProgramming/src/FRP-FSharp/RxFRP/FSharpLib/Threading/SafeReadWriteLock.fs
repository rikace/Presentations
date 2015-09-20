namespace Easj360FSharp

module SafeReadWriteLock =

    open System.Threading

    let readLock (rwlock : ReaderWriterLock) f  =
      rwlock.AcquireReaderLock(Timeout.Infinite)
      try
          f()
      finally
          rwlock.ReleaseReaderLock()

    let writeLock (rwlock : ReaderWriterLock) f  =
      rwlock.AcquireWriterLock(Timeout.Infinite)
      try
          f()
          Thread.MemoryBarrier()
      finally
          rwlock.ReleaseWriterLock()

    type MutablePair<'T,'U>(x:'T,y:'U) =
        let mutable currentX = x
        let mutable currentY = y
        let rwlock = new ReaderWriterLock()
        member p.Value =
            readLock rwlock (fun () ->
                (currentX,currentY))
        member p.Update(x,y) =
            writeLock rwlock (fun () ->
                currentX <- x;
                currentY <- y)


//    let readLock (rwlock : ReaderWriterLock) f  =
//      rwlock.AcquireReaderLock(Timeout.Infinite)
//      try
//        f() 
//      finally
//          rwlock.ReleaseReaderLock()
//    let writeLock (rwlock : ReaderWriterLock) f  =
//      rwlock.AcquireWriterLock(Timeout.Infinite)
//      try
//          f()
//          Thread.MemoryBarrier()
//      finally
//    rwlock.ReleaseWriterLock()
