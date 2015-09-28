var LanguagePrimitives__UnboxGeneric$String_String, LanguagePrimitives__UnboxGeneric$HubConnection_HubConnection_, ChatSignalRClient__signalR, ChatSignalRClient__serverHub, ChatSignalRClient__printResult$, ChatSignalRClient__onstart$, ChatSignalRClient__main$, ChatSignalRClient__log, ChatSignalRClient__j$, ChatSignalRClient__get_signalR$, ChatSignalRClient__get_serverHub$, ChatSignalRClient__get_log$;
  ChatSignalRClient__get_log$ = (function()
  {
    var objectArg = (window.console);
    return (function(arg00)
    {
      return (objectArg.log(arg00));
    });
  });
  ChatSignalRClient__get_serverHub$ = (function()
  {
    var conn = (ChatSignalRClient__signalR.hub);
    return conn;
  });
  ChatSignalRClient__get_signalR$ = (function()
  {
    return ((window.$).signalR);
  });
  ChatSignalRClient__j$ = (function(s)
  {
    return ((window.$)(s));
  });
  ChatSignalRClient__main$ = (function(unitVar0)
  {
    ((window.console).log("##Starting:## "));
    ((ChatSignalRClient__signalR.hub).url) = "http://localhost:48430/signalr/hubs";
    null;
    var client = (function(value)
    {
      return value;
    })(({}));
    var value = (function(msg)
    {
      return ChatSignalRClient__printResult$(msg);
    });
    ((client)["broadcastMessage"] = value);
    var conn = (ChatSignalRClient__signalR.hub);
    ((conn.createHubProxy("chatHub")).client = client);
    return ((ChatSignalRClient__signalR.hub).start((function()
    {
      return ChatSignalRClient__onstart$();
    })));
  });
  ChatSignalRClient__onstart$ = (function(unitVar0)
  {
    (function(value)
    {
      var ignored0 = value;
    })((ChatSignalRClient__j$("#joinChatBtn").click((function(_arg2)
    {
      var userName = (ChatSignalRClient__j$("#userName").val()).toString();
      var conn = LanguagePrimitives__UnboxGeneric$HubConnection_HubConnection_(ChatSignalRClient__serverHub);
      (function(value)
      {
        var ignored0 = value;
      })(((conn.createHubProxy("chatHub")).invoke("JoinChat", userName)));
      (function(value)
      {
        var ignored0 = value;
      })((ChatSignalRClient__j$("#chatDiv").show()));
      (function(value)
      {
        var ignored0 = value;
      })((ChatSignalRClient__j$("#userName").hide()));
      (function(value)
      {
        var ignored0 = value;
      })((ChatSignalRClient__j$("#joinChatBtn").hide()));
      (function(value)
      {
        var ignored0 = value;
      })((ChatSignalRClient__j$("#leaveChatBtn").show()));
      (function(value)
      {
        var ignored0 = value;
      })((ChatSignalRClient__j$("#leaveChatBtn").click((function(_arg1)
      {
        var _conn = LanguagePrimitives__UnboxGeneric$HubConnection_HubConnection_(ChatSignalRClient__serverHub);
        (function(value)
        {
          var ignored0 = value;
        })(((_conn.createHubProxy("chatHub")).invoke("LeaveChat", userName)));
        (function(value)
        {
          var ignored0 = value;
        })((ChatSignalRClient__j$("#chatDiv").hide()));
        (function(value)
        {
          var ignored0 = value;
        })((ChatSignalRClient__j$("#userName").show()));
        (function(value)
        {
          var ignored0 = value;
        })((ChatSignalRClient__j$("#leaveChatBtn").hide()));
        (function(value)
        {
          var ignored0 = value;
        })((ChatSignalRClient__j$("#joinChatBtn").show()));
      }))));
    }))));
    ChatSignalRClient__log("##Started!##");
    (function(value)
    {
      var ignored0 = value;
    })((ChatSignalRClient__j$("#submit").click((function(_arg3)
    {
      var text = LanguagePrimitives__UnboxGeneric$String_String((ChatSignalRClient__j$("#source").val()));
      var conn = LanguagePrimitives__UnboxGeneric$HubConnection_HubConnection_(ChatSignalRClient__serverHub);
      (function(value)
      {
        var ignored0 = value;
      })(((conn.createHubProxy("chatHub")).invoke("SendMessage", text)));
    }))));
    return ChatSignalRClient__log("##Sent MEssage!##");
  });
  ChatSignalRClient__printResult$ = (function(value)
  {
    var objectArg = ChatSignalRClient__j$("#results");
    return (function(_value)
    {
      var ignored0 = _value;
    })((function(arg00)
    {
      return (objectArg.append(arg00));
    })(((("\u003cp\u003e" + value) + "") + "\u003c/p\u003e")));
  });
  LanguagePrimitives__UnboxGeneric$HubConnection_HubConnection_ = (function(x)
  {
    return x;;
  });
  LanguagePrimitives__UnboxGeneric$String_String = (function(x)
  {
    return x;;
  });
  ChatSignalRClient__signalR = ChatSignalRClient__get_signalR$();
  ChatSignalRClient__serverHub = ChatSignalRClient__get_serverHub$();
  ChatSignalRClient__log = ChatSignalRClient__get_log$();
  ChatSignalRClient__main$()