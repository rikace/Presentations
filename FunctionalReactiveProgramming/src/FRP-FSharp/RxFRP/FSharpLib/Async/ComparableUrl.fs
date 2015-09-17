module ComparableUrl 

        open System

        type ComparableUri(uri: System.Uri) =
            inherit System.Uri(uri.AbsoluteUri)
    
            let elts (uri: System.Uri) =
                uri.Scheme, uri.Host, uri.Port, uri.Segments
    
            interface System.IComparable with
                member this.CompareTo(uri2) =
                compare (elts this) (elts(uri2 :?> ComparableUri))
    
            override this.Equals(uri2) =
                compare this (uri2 :?> ComparableUri) = 0