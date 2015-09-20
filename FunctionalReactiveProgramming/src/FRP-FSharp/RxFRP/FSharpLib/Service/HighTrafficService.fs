namespace Easj360FSharp

open System
open System.ServiceModel
open System.Collections.Generic
open System.Collections.Concurrent

module ServiceModule =

    [<ServiceContractAttribute>]
    type IHighTrafficCallbackService =
        [<OperationContract(IsOneWay=true, Name="Callback")>] 
        abstract Callback: Message:string -> unit

    [<ServiceContractAttribute(CallbackContract=typeof<IHighTrafficCallbackService>)>]
    type IHighTrafficService =
        [<OperationContract(IsOneWay=true, Name="Subscribe")>] 
        abstract Subscribe: Id:Guid -> unit

        [<OperationContract(IsOneWay=true, Name="Unsubscribe")>] 
        abstract Unsubscribe: Id:Guid -> unit

        [<OperationContract(IsOneWay=true, Name="Broadcast")>] 
        abstract Broadcast: Id:Guid * Message:string -> unit

    type TrafficRecord = {Id:Guid; Message:string}
    
    [<ServiceBehavior(Name="HighTrafficService", InstanceContextMode=InstanceContextMode.Single,  ConcurrencyMode=ConcurrencyMode.Multiple)>]
    type HighTrafficService() =
        
        [<DefaultValue>] 
        static val mutable private callbackList : ConcurrentDictionary<Guid,IHighTrafficCallbackService>

        static member private CallbackList
            with get() = if HighTrafficService.callbackList = null then                            
                                    if HighTrafficService.callbackList = null then
                                        HighTrafficService.callbackList <- new ConcurrentDictionary<Guid,IHighTrafficCallbackService>()
                         HighTrafficService.callbackList
                

        member private s.Broadcaster = MailboxProcessor<TrafficRecord>.Start(fun mbx ->
            let rec loop n = async {
                let! msg = mbx.Receive()
                let arrItems = HighTrafficService.CallbackList.ToArray()
                Array.Parallel.iter((fun (f:KeyValuePair<Guid,IHighTrafficCallbackService>) -> 
                                            if not(f.Key.Equals(msg.Id)) then
                                                f.Value.Callback(msg.Message)
                                    )) arrItems
                return! loop(n+1)
            }
            loop 0)


        interface IHighTrafficService with
            member s.Subscribe(value:Guid):unit =        
                    let cb = OperationContext.Current.GetCallbackChannel<IHighTrafficCallbackService>()
                    ignore( HighTrafficService.CallbackList.TryAdd(value, cb) )
              
            member s.Unsubscribe(value:Guid):unit =
                if HighTrafficService.CallbackList.ContainsKey(value) then                    
                    let cb = ref( OperationContext.Current.GetCallbackChannel<IHighTrafficCallbackService>() )
                    ignore( HighTrafficService.CallbackList.TryRemove(value, cb) )
                    

            member s.Broadcast(Id:Guid, message:string):unit =
                s.Broadcaster.Post({Id = Id; Message = message})


(*

    
    [<EntryPoint>]
    let main(args)=
        try 
            Console.WriteLine("Press ENTER to start the Service")
            Console.ReadLine()|>ignore
            let ep = new Uri("net.tcp://localhost:8099/ServiceModule")        
            let host = new ServiceHost(typeof<HighTrafficService>, ([| |] : Uri[] ) )        
            
            host.Open()
            Console.ReadLine()|>ignore
        with
        | ex -> let msg = ex.Message
                ()

        0

    <system.serviceModel>
        <behaviors>
            <serviceBehaviors>
                <behavior name="NewBehavior0">
                    <serviceMetadata />
                </behavior>
            </serviceBehaviors>
        </behaviors>
        <services>
            <service name="HighTrafficService.ServiceModule+HighTrafficService">
                <clear />
                <endpoint address="net.tcp://localhost:8099" binding="netTcpBinding"
                    name="TCP_Endpoint" contract="HighTrafficService.ServiceModule+IHighTrafficService"
                    listenUriMode="Explicit">
                
                </endpoint>
                <!--<endpoint address="net.tcp://localhost:8099/mex" binding="mexTcpBinding"
                    bindingConfiguration="" contract="IMetadataExchange" />-->
            </service>
        </services>
    </system.serviceModel>


*)