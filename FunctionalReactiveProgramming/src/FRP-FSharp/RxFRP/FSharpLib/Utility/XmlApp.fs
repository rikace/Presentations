namespace Easj360FSharp

//#r "System.Xml.dll"
//#r "System.Xml.Linq.dll"

open System.Xml.Linq

module XmlApp = 
    
        // Returns value of the specified attribute 
    let xattr s (el:XElement) = 
      el.Attribute(XName.Get(s)).Value
    // Returns child node with the specified name
    let xelem s (el:XContainer) = 
      el.Element(XName.Get(s))
    // Returns child elements with the specified name
    let xelems s (el:XContainer) = 
      el.Elements(XName.Get(s))
    // Returns the text inside the node
    let xvalue (el:XElement) = 
      el.Value

    // Return child node specified by a path
    let xnested path (el:XContainer) = 
      let res = path |> Seq.fold (fun xn s -> 
        // Upcast element to a container
        xn |> xelem s :> XContainer) el
      // Downcast the result back to an element
      res :?> XElement


