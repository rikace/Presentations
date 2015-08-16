#r "System.Runtime.Serialization"
#r "System.ServiceModel"
#r "FSharp.Data.TypeProviders"

open System
open System.ServiceModel
open Microsoft.FSharp.Linq
open Microsoft.FSharp.Data.TypeProviders

type TerraService = WsdlService<"http://msrmaps.com/TerraService2.asmx?WSDL">

let terraClient = TerraService.GetTerraServiceSoap ()
let myPlace = new TerraService.ServiceTypes.msrmaps.com.Place(City = "Redmond", State = "Washington", Country = "United States")
let myLocation = terraClient.ConvertPlaceToLonLatPt(myPlace)
printfn "Redmond Latitude: %f Longitude: %f" (myLocation.Lat) (myLocation.Lon)

