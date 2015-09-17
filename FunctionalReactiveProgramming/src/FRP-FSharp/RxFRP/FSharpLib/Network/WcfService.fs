namespace Easj360FSharp 

 module WcfService = 
    open System
    open System.IO
    open System.Runtime.Serialization
    open System.ServiceModel

    [<ServiceContract>]
    type IService = 
        [<OperationContract>]
        abstract GetData : value:int -> string
        
    type Service() =
        interface IService with 
            member x.GetData value =
                sprintf "%A" value


(*
#light

#I @"C:\Program Files\Reference Assemblies\Microsoft\Framework\v3.0";;
#r "System.ServiceModel.dll"
#r "System.Runtime.Serialization.dll"
open System
open System.Runtime.Serialization
open System.ServiceModel
open System.ServiceModel.Description

[<DataContract>] 
// specifies this class to serialize and deserialize using
// DataContractSerializer
type SimpleCompound()=
     let mutable x=0
     let mutable y=0
     [<DataMember>]
     // indicates to participate in WCF serialization 
     member o.X with get()=x and set(v)=x<-v
     [<DataMember>]
     member o.Y with get()=y and set(v)=y<-v

[<ServiceContract(Namespace="http://www.abraham-consulting.biz/FSharpWCFBasis")>]
// definition of WCF Service
type ISimpleService = interface
   // enables this method to participate in WCF Service
   [<OperationContract>]
   abstract Add: a:int*b:int ->int
   [<OperationContract>]
   abstract Random: unit-> SimpleCompound
end

[<ServiceBehavior(ConcurrencyMode = ConcurrencyMode.Multiple, InstanceContextMode = InstanceContextMode.Single)>]
//Specifies the internal execution behavior of a service contract implementation
type SimpleService()=
   interface ISimpleService with
      member o.Add(a,b)=a+b
      member o.Random() = let r = new System.Random()
                          let sc = new SimpleCompound()
                          sc.X<-r.Next(10)
                          sc.Y<-r.Next(23)
                          sc
   
   
let createServiceHost(wsUri:string,tcpUri:string)= 
      let uri = new Uri(wsUri)
      // Create a service host
      let sh = new ServiceHost((typeof<SimpleService>),[|uri|])
      // add web service channel
      sh.AddServiceEndpoint((typeof<ISimpleService>),new WSHttpBinding(),"ws")|>ignore
      // add tcp channel
      sh.AddServiceEndpoint((typeof<ISimpleService>),new NetTcpBinding(),tcpUri)|>ignore
      // enable metadata behaviour to create client proxy
      let smb = new ServiceMetadataBehavior()
      smb.HttpGetEnabled<-true
      sh.Description.Behaviors.Add(smb)
      sh
   




let servicehost = createServiceHost("http://localhost:2008/SimpleService","net.tcp://localhost:2009/SimpleService/tcp")
servicehost.Open()
// open visual studio 2008 Command prombt and type  wcftestclient http://localhost:2008/SimpleService
// to test service

//servicehost.Close()


*)