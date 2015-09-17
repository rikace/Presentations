namespace Easj360FSharp

module AsyncFileSeqGenerator =


  let ofAsync (stream:System.IO.Stream) =
    let count = 1024
    let buffer = Array.zeroCreate count
    let segment = ref (System.ArraySegment<_>())
    let bytesRead = ref -1

    let setSegment bytesRead = segment := System.ArraySegment<_>(buffer, 0, bytesRead)
    let handleError e = printfn "%A" e

    seq {
      while !bytesRead <> 0 do
        // Immediately processes the promise to create an ArraySegment containing the result.
        Async.StartWithContinuations(stream.AsyncRead(buffer, 0, count), setSegment, handleError, handleError)
        yield !segment
    }

