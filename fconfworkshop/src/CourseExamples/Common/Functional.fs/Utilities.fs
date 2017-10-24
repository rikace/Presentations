namespace FunctionalConcurrency

[<AutoOpen>]
module Utilities =
    open System.IO
    open System
    open AsyncEx

    let inline flip f a b = f b a 

    /// Given a value, apply a function to it, ignore the result, then return the original value.
    let inline tee fn x = fn x |> ignore; x


    type BaseStream(stream:Stream) =
        member this.AsyncWriteBytes (bytes : byte []) = async {
                do! stream.AsyncWrite(BitConverter.GetBytes bytes.Length, 0, 4)
                do! stream.AsyncWrite(bytes, 0, bytes.Length)
                return! stream.FlushAsync() 
            }

        member this.AsyncReadBytes(length : int) =
            let rec readSegment buf offset remaining = async {
                    let! read = stream.AsyncRead(buf, offset, remaining)
                    if read < remaining then
                        return! readSegment buf (offset + read) (remaining - read)
                    else
                        return () }
            async {
                let bytes = Array.zeroCreate<byte> length
                do! readSegment bytes 0 length
                return bytes
            }

        member this.AsyncReadBytes() = async {
                let! lengthArr = this.AsyncReadBytes 4
                let length = BitConverter.ToInt32(lengthArr, 0)
                return! this.AsyncReadBytes length
            }