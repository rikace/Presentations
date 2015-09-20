#if INTERACTIVE
#r "FSharp.PowerPack.dll"
#endif

namespace Easj360FSharp

open System.IO
open System.Net
open System.Text

[<AutoOpen>]
module WebRequestExtensions =
  type System.Net.WebRequest with
    member x.AsyncGetRequestStream() = 
      Async.FromBeginEnd(x.BeginGetRequestStream, x.EndGetRequestStream)

module testWebExtesnion  = 

    let myAppId = "appid"
    let detectUri = "http://api.microsofttranslator.com/V1/Http.svc/Detect?appId=" + myAppId
    let translateUri = "http://api.microsofttranslator.com/V1/Http.svc/Translate?appId=" + myAppId + "&"
    let languageUri = "http://api.microsofttranslator.com/V1/Http.svc/GetLanguages?appId=" + myAppId
    let languageNameUri = "http://api.microsofttranslator.com/V1/Http.svc/GetLanguageNames?appId=" + myAppId

    let getStreamData (uri:string) =
      async { let request = WebRequest.Create languageUri 
              use! response = request.AsyncGetResponse()
          
              return [use stream = response.GetResponseStream()
                      use reader = new StreamReader(stream)
                      while not reader.EndOfStream do yield reader.ReadLine()] }

    let languages = Async.RunSynchronously (getStreamData languageUri)
    let languagesWithLocale locale = Async.RunSynchronously (getStreamData <| sprintf "%s&locale=%s" languageNameUri locale)

    let callService (text:string) (uri:string) =
      async { // Must be POST to include translation text
              let request = WebRequest.Create uri :?> HttpWebRequest
              do request.Method <- "POST"
              do request.ContentType <- "text/plain"
          
              // Insert text into body
              let bytes = Encoding.UTF8.GetBytes text
              let contentLength = Array.length bytes
              do request.ContentLength <- int64 contentLength
              use! os = request.AsyncGetRequestStream()
              do! os.AsyncWrite(bytes, 0, contentLength)
          
              use! response = request.AsyncGetResponse()
              use stream = response.GetResponseStream()
              use reader = new StreamReader(stream)
              return! reader.AsyncReadToEnd() }

    let translateText text toLanguage =
      async { let! fromLanguage = callService text detectUri
              let uri = sprintf "%sfrom=%s&to=%s" translateUri fromLanguage toLanguage
              let! translated = callService text uri
              return (fromLanguage, toLanguage, text, translated) }
          
    let text = "Hello TechReady, are you ready to learn F#?"

    //let textResults = 
    //  Async.RunSynchronously 
    //    (Async.Parallel 
    //      [for language in languages.Item -> translateText text language])
    //  
