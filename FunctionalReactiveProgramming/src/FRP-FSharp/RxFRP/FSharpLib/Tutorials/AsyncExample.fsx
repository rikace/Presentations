#r "System.Net.dll"
(* 
Esimerkki perustuu Tomas Petricekic koodifragmenttiin, joka löytyy osoitteesta http://fssnip.net/6e

Esimerkin idea on demonstroida seuraavia ominaisuuksia:
1) Luokkien laajentaminen ja hyödyntäminen ilman periyttämistä tai muita vastaavia raskaita ratkaisuja. 
2) Async, AsyncBuilder ja niiden käyttö F#:ssa
*)
module HttpListenerHelpers = 
    open System
    open System.IO
    open System.Net
    open System.Threading
    open System.Collections.Generic

    // Alla HttpListener ja WebClient luokkiin laajennetaan tuki F#:n Async tyypille. 
    //
    // Näistä WebClient luokan laajennus on vaikeaselkoisempi. Olennaisesti Async.FromContinuations 
    // tuottaa Async-luokasta sellaisen instanssin joka kuuntelee kolmea asynkronista tapahtumaa:
    // - Tuli vastaus yritykseen ladata url. Ei ongelmia.
    // - Jotain meni pieleen.
    // - Lataus peruutettiin.
    // 
    // Async-luokka toimii eräänlaisena proxynä WebClientin DownloadDataComplited tapahtumille. 
    //
    // Ensimmäisessä tapauksessa luodaan Async proxy HttpListenerin BeginGetContext ja EndGetContext delegaatille. 
    // Käytännössä tämä sallii sivukyselyn käsittelyn synkronisesti ja vastuksen kirjoittamisen HttpResponsen 
    // tulostevirtaan kunhan sellainen WebClientilta saapuu (ks. HttpListenerin StartMetodi).
    type System.Net.HttpListener with
        member x.AsyncGetContext() = 
            Async.FromBeginEnd(x.BeginGetContext, x.EndGetContext)

    type System.Net.WebClient with        
        member x.AsyncDownloadData(uri) = 
            Async.FromContinuations(fun (cont, econt, ccont) ->
                x.DownloadDataCompleted.Add(fun res ->
                    if res.Error <> null then econt res.Error
                    elif res.Cancelled then ccont (new OperationCanceledException())
                    else cont res.Result)
                x.DownloadDataAsync(uri) )

    // C#:n extension metodeilla pystyy laajentamaan luokalle vain instanssin metodeja muttei ei staattisia metodeja. 
    // F#:ssa ei ole tätä rajoitusta.
    type System.Net.HttpListener with 
        static member Start(url, f) = 
            let tokenSource = new CancellationTokenSource()
            
            // Asycn Start käynnistää uuden säikeen. Säikeen sisällä käynnistetään http-liikenteen kuuntelija parametrina 
            // annettuun osoitteeseen.
            //
            // Huomaa. Async on täysin eri asia kuin async-avainsana ja siihen liittyvä AsyncBuilder luokka. Async-luokka 
            // sisältää logiikan laskennan suorittamiseen asynkronisesti. AsyncBuilderilla taas luodaan asynkronisissa 
            // työnkulkuja (hyödyntäen Async-objekteja). Siten se on abstraktio tasoltaan asteen Asyncin yläpuolella.
            Async.Start
                ((async { 
                    use listener = new HttpListener()
                    listener.Prefixes.Add(url)
                    listener.Start()
                    // Tässä odotetaan loopissa niin pitkään että tulee kysely. "let! context = " blokkaa etenemisen loopissa (tietenkään blokkaamatta koko sovellusta).
                    // Kun kysely lopulta saapuu, se passataan saman tein toiseen säikeeseen käsiteltäväksi ja pyörähdetään odottamaan seuraavaa kyselyä.
                    // Kannattaa huomata, että tässä tapahtumien kuuntelu ja kirjaaminen kuuntavaksi määritetään olemattomalla määrällä koodia. 
                    // AsyncGetContext() "putkittaa" HttpListener luokan BeginGetContext ja EndGetContext tapahtumat AsyncBuilder luokalle. 
                    
                    // "let!" on syntaktista sokeria, joka kääntyy tässä AsyncBuilder-instanssin Bind -metodi kutsuksi seuraavaan tapaan:
                    // async.Bind(listener.AsyncGetContext(), fun(context) -> Async.Start(f context, tokenSource.Token)})
                    //
                    // Huomaa että tässä cancellationToken on Async.Start:n käynnistämän säikeen lopetuskahva. HttpListener-objektin Close() 
                    // metodia ei tarvitse kutsua, koska säikeen lopettaminen tekee olennaisesti saman ja hieman enemmänkin. 
                    while true do 
                        let! context = listener.AsyncGetContext()
                        Async.Start(f context, tokenSource.Token)}),
                    cancellationToken = tokenSource.Token)
            tokenSource

        // Tämä toimi kuten Start yllä, mutta pyyntö käsitellään synkronisesti ja seuraava käsitellään vasta kun ensimmäinen on käsitelty kokonaan.
        static member StartSynchronous(url, f) =
            HttpListener.Start(url, f >> async.Return) 
            // Sivuhuomiona: ">>" operaattori on F#-versio matematiikan yhdistetyn funktion pallo-operaattorille; 
            // esim f o g (x) on sama kuin g(f(x)) olettaen, että f on kuvaus x->y ja g kuvaus y->z 
            // f >> async.Return on sama kuin (fun context -> async.Return ( f (context))). 04-MoreAdvanceStuff

module ASyncLister = 
    open System
    open System.IO
    open System.Net
    open HttpListenerHelpers
    let mutable token = null
    let mutable mirrorRoot = null
    
    let getProxyUrl (ctx:HttpListenerContext) = 
        Uri(mirrorRoot + ctx.Request.Url.PathAndQuery)
    
    // Funktio käsittelee poikkeukset asynkronisesti
    // Tämä funktio kääntyy muotoon 
    // [1] AsyncBuilder.Delay(fun () -> 
    //    use wr = new StreamWriter(ctx.Response.OutputStream)
    //     [clip]
    //   ctx.Response.Close()
    // Kun tapahtuu virhe:
    // do! asyncHandleError ctx e 
    // kääntyy muotoon 
    //
    // AsyncBuilder.Bind([1], [seurava async operaatio]) (ks. alla)
    //
    // Tässä Bind-metodi suorittaa operaation, jonka Delay on lykännyt. 
    let asyncHandleError (ctx:HttpListenerContext) (e:exn) = async {
       use wr = new StreamWriter(ctx.Response.OutputStream)
       wr.Write("<h1>Request Failed</h1>")
       wr.Write("<p>" + e.Message + "</p>")
       ctx.Response.Close() }

    // AsyncBuilder mahdollistaa asynkronisten operaatioiden putkittamisen 
    // minimi määrällä pään rapsutusta. Tässä kokonaisuudessa putkitetaan kolme
    // asynkronista operaatiota. 
    // 1) Sivu pyyntö tulee kun on tullakseen. Kun ase tulee aloitetaan tämän koodi blokin suoritus.
    // 2) Ladataan asynkronisesti data verkosta.
    // 3a) kun lataus byte[] on käytettävissä kirjoitetaan data asynkronisesti kuunneltavan kontekstin 
    // tulostevirtaan 
    // 3b) jos homma meni kaikki meni pieleen niin kirjoitetaan virhe ja tuli poikkeus. Kirjoitetaan virhe tulosvirtaan.
    // "let! data = [...]" ja "do! ctx.Response.OutputStream.AsyncWrite(data)" putkittaa operaatio automaattisesti siten että 
    // jälkimmäinen suoritetaan vasta kun ensimmäinen palauttaa arvon. Sovellus ei jää odottamaan että näin tapahtuu.
    // saman voisi toteuttaa myös ilman syntaktista sokeria ja lopputulos näyttäisi tämän tapaiselta.
    // [...]
    // async.Delay(fun () ->
    //    let wc = new WebClient()
    //    try
    //      async.Bind(wc.AsyncDownloadData(getProxyUrl(ctx)),(fun data ->
    //           async.Bind(ctx.Response.OutputStream.AsyncWrite(data),(fun () ->
    //                  async.Return()))
    //    catch e -> 
    //         async.Bind(asyncHandleError ctx e, async1.Return())
    //
    // F#:n async-laskentailmaus (computational expression) tekee asynkronisesta ohjelmoinnista huomattavasti 
    // helpompaa kuin mitä olisi monilla muilla kielillä tekemällä asynkronisesti suoritettavasta koodista huomattavasti 
    // helppolukuisempaa ja tiiviimpää.
    let asyncHandleRequest (ctx:HttpListenerContext) = 
        async {
            let wc = new WebClient()
            try
                let! data = wc.AsyncDownloadData(getProxyUrl(ctx))
                do! ctx.Response.OutputStream.AsyncWrite(data) 
                do! async{ctx.Response.OutputStream.Close()}
            with e ->
                do! asyncHandleError ctx e 
        }


    /// Tämä käynnistää http liikenteen kuuntelijan. Muista sulkea se funktiolla Stop. Toista kuuntelijaa ei voi käynnistää samaan porttiin. 
    /// (Asiaan mitenkään liittymättä: kolmea kauttaviivaa '///' voi käyttää metodien dokumentoimiseen. Kun hiiren vie alla olevan metodin päälle näkee tämän rivin.)
    let StartMirroring (url) =
        mirrorRoot <- url
        token <- HttpListener.Start("http://localhost:8080/", asyncHandleRequest)
    
    let Stop () = 
        if (token <> null) then token.Cancel()

module SyncVersion =
    // Esimerkin vuoksi sama logiikka synkronisena.
    open System
    open System.IO
    open System.Net
    open HttpListenerHelpers
    let mutable token = null
    let mutable mirrorRoot = null
    
    let getProxyUrl (ctx:HttpListenerContext) = 
        Uri(mirrorRoot + ctx.Request.Url.PathAndQuery)

    let handleError (ctx:HttpListenerContext) (e:exn) =
       use wr = new StreamWriter(ctx.Response.OutputStream)
       wr.Write("<h1>Request Failed</h1>")
       wr.Write("<p>" + e.Message + "</p>")
       ctx.Response.Close()

    let handleRequest (ctx:HttpListenerContext) =
        let wc = new WebClient()
        try
          let data = wc.DownloadData(getProxyUrl(ctx))
          ctx.Response.OutputStream.Write(data, 0, data.Length)
          ctx.Response.Close()
        with e ->
         handleError ctx e
 
    let StartMirroring (url) =
        mirrorRoot <- url
        token <- HttpListener.StartSynchronous("http://localhost:8080/", handleRequest)
    let Stop () = 
        if (token <> null) then token.Cancel()

(*

// Käynnistä peilaus suorittamalla jompikumpi alla olevista StartMirroring metodeista.
ASyncLister.StartMirroring "http://msdn.microsoft.com"
ASyncLister.Stop()

SyncVersion.StartMirroring "http://msdn.microsoft.com"
SyncVersion.Stop()
*)