namespace Easj360FSharp

open System
open System.Net
open System.IO
open System.Text

module WebClientOutput =

    type WebClientInput =
        | StringInput of String
        | UriInput of Uri
    
    type WebClientOutput =
        | MalformedUri of string
        | TextContent of string 
        | BinaryContent of byte []
        | NoContent
        | WebClientException of WebException
    
    type UriOutput =
    | Uri of Uri
    | Malformed of string

    let buildUri stringUri =
        try Uri( new Uri(stringUri) )
        with | _ -> Malformed(stringUri)

    let downloadWithWebClient (inputUri: WebClientInput) =
        let downloadFromUri (uri: Uri) =    
            try 
                use client = new WebClient()
                let dlData = client.DownloadData(uri)
                if dlData.Length = 0 then NoContent
                else if (client.ResponseHeaders.["content-type"]
                               .StartsWith(@"text/")) 
                    then
                        let dlText = Encoding.Default.GetString(dlData)
                        TextContent(dlText)
                    else
                        BinaryContent(dlData)
            with
               | :? WebException as e -> WebClientException(e)
        match inputUri with
        | UriInput(classUri) -> downloadFromUri classUri
        | StringInput(stringUri) ->
            match buildUri stringUri with
            | Uri(s) -> downloadFromUri s
            | Malformed(s) -> MalformedUri(s)

    let printWebClientOutput clientOutput =
        match clientOutput with
        | MalformedUri(uri) -> printfn "Input Uri was malformed: %s" uri
        | TextContent(content) -> printfn "Page Content: %s" content
        | BinaryContent(content) -> printfn "Binary Data: %d" content.Length
        | NoContent -> printfn "No content was found."
        | WebClientException(e) -> printfn "Exception: %s" (e.ToString())

    let downloadToFile (inputUri: WebClientInput) outputLocation =
        match downloadWithWebClient inputUri with 
        | TextContent(text) -> File.WriteAllText( outputLocation, text )
        | BinaryContent(binary) -> File.WriteAllBytes( outputLocation, binary )
        | _ -> printfn "Download Failed"


