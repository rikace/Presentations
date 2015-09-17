namespace Easj360FSharp

open System
open System.IO
open System.Security.Cryptography

module Hashing =

    // synchronous file hashing
    let hashFile f = 
        use fs = new FileStream(f, FileMode.Open)
        use hashFunction = new SHA512Managed()
        hashFunction.ComputeHash fs |> Convert.ToBase64String

    // async stream hashing
    let hashAsync bufferSize (hashFunction: HashAlgorithm) (stream: Stream) =
        let rec hashBlock currentBlock count = async {
            let buffer = Array.zeroCreate<byte> bufferSize
            let! readCount = stream.AsyncRead buffer
            if readCount = 0 then
                hashFunction.TransformFinalBlock(currentBlock, 0, count) |> ignore
            else 
                hashFunction.TransformBlock(currentBlock, 0, count, currentBlock, 0) |> ignore
                return! hashBlock buffer readCount
        }
        async {
            let buffer = Array.zeroCreate<byte> bufferSize
            let! readCount = stream.AsyncRead buffer
            do! hashBlock buffer readCount
            return hashFunction.Hash |> Convert.ToBase64String
        }

    // async file hashing
    let hashFileAsync f =    
        async {
            use fs = File.OpenRead f
            use hashFunction = new SHA512Managed()
            return! hashAsync (*bufferSize*) (int fs.Length) hashFunction fs
        }





