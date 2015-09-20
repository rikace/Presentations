namespace FSharpTarinin.Basics
module M1_HelloFSharp = 
     (*
     1. Mikä on F# ja miksi käyttäisin sitä?

    Mitä F# on?
        * F# on multiparadigma-ohjelmointikieli, jonka painopiste on funktionaalisessa ohjelmointiparadigmassa.
        * F# on yksi kolmesta Visual Studion mukana tulevista kielestä. 
        * F#:n juuret ovat OCaml ja ML-kielessä. Myös Haskell on keskeinen vaikute. (Ks. http://www.cs.helsinki.fi/u/poikelin/FSharp/taustaa.html)
    
    Miksi?
        - F# on ilmaisuvoimainen ja syntaksiltaan tiivis kieli
            * Vaikka F# on vahvasti tyypitetty kieli, sen monet rakenteet muistuttavat keveydessään 
              dynaamisesti tyypitettyjä skripti kieliä (kuten Python ja Ruby). Interactive-ympäristön 
              (REPL-loop) avulla kehittäjä pääsee keskittymään ongelman ytimeen samalla kun 
              testaa koodiaan.
            * F# koodin rivimäärä on usein 2-5 alhaisempi kuin vastaavan C#-ratkaisun. Koodin luettavuus 
              on samaa tasoa kuin C#- tai Java-koodin - joskin eri asiat ovat helppolukuisia/helppolukuisia. 
              F#:ssa on enemmän kryptisiä lyhenteitä ja operaattoreja, siinä missä Javassa ja C#:ssa on enemmän 
              avainsanoja ja idiomeja muistavaksi.
            * F# on abstraktiotasoltaan korkeampi kieli kuin C#. Vastaavasti kuin C# on abstraktiotasoltaan korkeampi 
              kieli kuin CIL/assembler.
            * F# ohjaa oletuksena sivuvaikutuksettomaan koodaukseen. Muutoinkin F# antaa huomattavasti näkemystä 
              siitä *miksi* tehdä asioita myös muissa kielissä jollain tavalla. 
        - F#:lla on C:n suorituskykyprofiili.  Toisin sanoen se on vain hieman (=selvästi alle kymmenen kertaa) hitaampi 
            kuin hyvin optimoitu C-koodi, ja siten suunnilleen yhtä tehokas kuin Java ja C#. Dynaamiset, skriptikielet 
            (kuten PHP, Python, ja Ruby) ovat 10-1000-kertaa optimoitua C-koodia hitaampia. Ero korostuu 
            laskentaintensiivisissä tehtävissä.
        - Kaikki .NET-frameworkin luokkakirjastot ovat käytettävissä. F# on .NET-kieli ja kääntyy CIL:ksi (kuten C# ja 
            VB.NET; CIL = Common intermediate language; AKA Microsoft Intermediate lanuguage (MSIL)).

    Missä? 
        - F# soveltuu parhaiten Business-logiikan, monimutkaisten algoritmien ja datan käsittelyn toteuttamiseen. 
            (Ks. Don Symen esitys F# 3.0: data, services, Web, cloud, at your fingertips http://channel9.msdn.com/Events/BUILD/BUILD2011/SAC-904T)
        - F#:lla voi toteuttaa myös käyttöliittymäkerroksen ja tiedonesityskerroksen. WebSharper on kiinnostava esimerkki siitä, kuinka käyttöliittymän voi toteuttaa 
            funktionaalisen ohjelmointi paradigman hengessä (http://websharper.com/home).
     
    Käytännössä?
        - Tämä esitys on laadittu mahdollisimman käytännön läheiseksi sukellukseksi F#:n syntaksiin. Suurin osa esityksestä 
          onkin koodia.
        - Helpoin tapa suorittaa alla oleva koodi, ja katsoa mitä tapahtuu, on mennä ensimmäiselle koodiriville ja painaa Alt-ä.
          Alt-ä suorittaa vain yhden rivin ja usein ilmaus jatkuu toisella rivillä. Tälläin maalaat ja suoritat koodi 
          alt-enter-yhdistelmällä. Huom. ota rivien alut mukaan, sillä sisennyksellä *on* F#:ssa merkitystä. 
          (Valikoiden kautta saa aikaan saman.)
     
    Esim. siirry alle olevalle riville "System.Console.Beep ()" ja paina alt+ä. Jos äänet ovat päällä pitäisi kuulua "beep".
    *)
     System.Console.Beep()



     // Alla oleva merkkijono tulostuu F#-interaktiven tulosjonoon muodossa: 
     // val it : string = "Hello world"'
     "Hello world"

     // Klassinen hello world teksti ikkunassa.
     open System.Windows.Forms
     let form = new Form()
     form.Controls.Add(new Label(Text = "Hello world!"))
     form.Show()

     // Klassinen "Hello, nimi!" muunnos Hello Worldista.
     let form2 = new Form()
     let question = new Label(Dock = DockStyle.Top, Text = "Kuka olet?") 
     let namefield = new TextBox(Dock = DockStyle.Top);
     let hello = new Label(Dock = DockStyle.Top); 
     namefield.KeyUp.Add(fun e-> hello.Text <- match namefield.Text with "" -> "" | text -> "Hello, " + text + "!")
     form2.Controls.AddRange [|hello; namefield; question|]
     form2.Show()

     // F# interactiven sisällä voi muokata luotu lomaketta "lennossa".
     let beep () = for i = 0 to 2 do System.Console.Beep()
     namefield.KeyUp.Add(fun e-> hello.Text <- match namefield.Text with | "" -> "" | "test" | "Test" -> beep (); "Syötä oikea nimi!" | text -> "Hello, " + text + "!")

module M2_LiteraalitJaTunnisteet = 
    (*
    2. Tunnisteet, primitiivityypit ja muutettavuus
    *)

    // F# tunnisteet esitellään let-avainsanalla. 
    // Tunnisteen tyypin näkee viemällä hiiren kursori nimen päälle 
    // Primitiivi tyyppien literaalit ovat pääsääntöisesti samat kuin C#:ssa.
    let x = 1 // 32 bittinen kokonaisluku
    // tunniteen tyypin voi määittää syntaksilla let nimi : tyyppi
    let x2 : System.Int64 = 1L // 64 bittinen kokonaisluku 
    let y : float = 1.0  // System.Double, 64 bittinen liukuluku; C#:ssa double, F# float (HUOM!)
    let y3 = 1.0f // System.Single, 32 bittinen liukuluku; C#:ssa float, F#:ssa float32 (HUOM!)
    let str = "merkkijono"
    let chr = 'm'
    
    // Välimerkkejä voi käyttää, jos haluaa pitkiä muuttujia. Tosin suomenkielisessä näppiksessä 
    // ` (backtick)löytyy hankalasti yhdistelmällä shift-´-[jokin merkki]:
    let ``tämä tunniste vaatii välimerkkejä ja on pitkä, mutta selkeä`` = "juttu"
    ``tämä tunniste vaatii välimerkkejä ja on pitkä, mutta selkeä``

    // Toisin kuin C#:ssa oletuksena tunnisteet eivät ole muutettavissa (tunniste on ei-muutettava ("immutable")).
    // Muuttujien (muutettavien tunnisteiden) esittely pitää tehdä explisiittisesti käyttäen mutable-avainsanaa.
    // (C#:n oletus arvo on päin vastainen, ei-muutettavan tunnisteen esittelyssä pitää käyttää readonly-avainsanaa.)
    let mutable z = 1
    // Muuttujaan sijoitus tapahtuu '<-' -operaattorilla
    z <- 5
    z
    // Yhtäsuuruus merkki ei toimi koskaan (!!!) muuttujaan sijoituksena:
    z = 1 // palauttaa arvon true. x:n arvo ei muutu.
    
    // Matematiikan tapaan yhtäsuuruutta käytetään vain seuraavassa kahdessa merkityksessä:
    // 1) uuden tunnisteen/nimen esittelyyn
    // 2) yhtäsuuruus vertailuissa

    // Tämä voi tuntua aluksi hämmentävältä. Toisaalta: ohjelmistojen bugit johtuvat usein siitä, 
    // että jokin muuttuja ei ole siinä tilassa missä oletetaan. Kun laskentaa aletaan tehdä hajautettusti 
    // usealla ytimellä ongelma räjähtää helposti käsiin. Tässä mielessä ei-muokattavissa olevat tunnisteet 
    // on parempi oletusarvo kuin se, että tunnisteet ovat oletuksena muutettavia.

    // Ei-muutettavuus ei C# näkökulmasta tarkoita sitä, että kyseessä olisi vakio. Tunnisteen arvoa ei leivota sisään
    // luokkakirjaston CIL-koodiin; sitä ei vaan voi muuttaa sen jälkeen kun se on kerran asetettu. 
           
    // Itse asissa tarkasti ottaen F# koodissa ei juurikaan käytetä C#-vakiota vastaavaa rakennetta. C#:n:
    // public const int Vakio = 1;
    // on F#:ksi
    [<Literal>]     
    let Vakio = 1
    // Yleistäen ja mutkia oikoen: muuttuja = muutettavissaoleva tunniste (mutable identifier) ja vakio = ei-muutettava 
    // tunniste (immutable identifier). Ei-muutettava tyyppi (immutable type) on sellainen jonka instanssien tilaa ei 
    // voi muuttaa. Esim. primitiivi tyypit, string ja DateTime. Muutettavissaolevat tyyppi (mutable type) on 
    // sellainen, jonka instanssien tila voi muokata (suurin osa .NET:n Frameworkin luokista). 
    // Siis, ohjelmoijan näkökulmasta seuraava on käytännössä vakio:
    let Vakio2 = 1 
                                                                                                    
module M3_Listat =     
    // Listat ja taulukot
    // Toisin kuin C#;ssa perus listat ja taulukot ovat muuttumattomia (immutable). 
    let list = [1;2;3]
    let array = [|1;2;3|] 
    
    // list on F# spesifi linkitettynä listana toteutetta rakenne. Array on C#:n System.Array

    // Huom. vaikka itse array tunniste on merkitty ei-muokattavaksi, System.Array on muokattava tyyppi.
    // Seuraava on sallittua
    array.[0]<-5
    array
    // Se että tunnisteen asettaa ei-muutettavaksi tarkoittaa vain sitä, että po tunnisteen arvo ei voi muuttua: 
    // array <- [|1;2;3|]    // Virhe: This value is not mutable

    // Listan indeksointi on mahdollista
    list.[0]
    // Mutta muokkaaminen ei; ks. alla oleva, kääntäjän takia pois kommentoitu virheentuottava esimerkki. 
    // list.[1] <- 5  // Virhe: "Property Item cannot be set"

    // .NET:n muutettavissa olevia rakenteita voi toki myös käyttää ja ne toimivat kuten C#:ssa. 
    let genericList = System.Collections.Generic.List<string>()

    // Huom. F# list ja List<T> ovat täysin eri rakenteita:
    //               Immutable items  Immutable list  Element lookup
    //F# list        Kyllä!           Kyllä!          O(n), eli O(1) listan alkuun
    //Array          Ei               Kyllä!          O(1)
    //Generic List   Ei               Ei              O(1)

    // Best practice on käyttää listoja ei-muutettavina rakenteina (immutableina) ja olla viittaamatta 
    // suoraan indekseihin. Sen sijaan että poistat listasta alkion, tee uusi lista, josta on suodatettu 
    // poistettava alkio.
                                                                                           
    // Kolmas listaa muistuttava rakenne on monikko (tuple).
    // Siinä missä listat ovat n kappaletta yhtä tyyppiä, niin monikko (tuple) on yksi kappale n:ää tyyppiä:
    let tuple = (1,"a",0.4) 
    let tuple2 = 1,"a"  // Sulut ovat vapaaehtoisia! 
    
    // Ja monikon (tuplen) voi purkaa kivasti:
    let eka, toka = tuple2
    // Useat muuttujam esittely samalla rivillä hyödyntää monikkoa:
    let a, b, c = "a", "b", "c" 

module M4_Funktiot = 
    (*
    3. Funktiot

    Teemat
     - funktiot ovat ensimmäisen luokan kansalaisia
     - Yhdistetty funktio
     - currying
     - rekursio
    *)

    // Funktio on muuttuja siinä missä string. 
    // Funktion voi määritellä käyttäen lamda-ilmausta
    let f = fun x -> x
    // Tai esittelemällä parmetrit heti nimen jälkeen
    let f2 x = x
    // kaikilla funktioilla on vähintään yksi parametri, mutta ainokainen parametri voi olla tyyppi "ei mitään". 
    // Ei mitään tyypin nimi on 'unit' (Microsoft.FSharp.Core.Unit) ja sen literaali on (). 
    // Ei mitään on objekti toisin kuin C#:n tai Javan null. 
    let nothingh = ()
    // C#:n void metodien palautus arvo on F#:ssa unit samoin kuin Action delekaattien. Vastaavasti parametrittomien 
    // metodien parametrina on unit.
    let Action x = ()  // val Action : 'a -> unit
    Action 1
    let NoParameters () = 1
    NoParameters // palauttaa funktion tyyppi unit -> int  
                 // Tunnisteen arvo vaihtelee suorituskerroittain. Tätä kirjoittaessa se on <fun:it@197-30>. 
                 // Tässä suhteessa function esittely ei eroa int32 tyyppisen tunnisteen esittelystä.
    NoParameters () // palauttaa 1
    
    // Alla olevan plus-funktion tyyppi on int->int->int
    let plus x y = x + y
    plus 1 2

    // Yksi kiinnostavimmista ominaisuuksista on funktioiden ketjutus
    // Idean hahmottaa helpoiten sulkeistamalla. Kun ensimmäisen int:n korvaa 5:llä jää tyypiksi int->int 
    // (itse mielessäni usein sulkeistan tähän malliin int->(int->(int)). Kun sijoitan ensimmäisen "slottiin" 
    // numeron 5 (5->(int->(int)), saan funktion int->int kuin sijoitan seuraavan slottiin 5 saan kokonaisluvun 
    // tyyppiä int.
    let lisää_viiteen = plus 5

    // Sijoittamalla ensimmäiseksi arvoksi 5 funktioon tyyppiä int->int, jäljelle jää int. Funkition varsinainen koodi suoritetaan vasta tässä:
    lisää_viiteen 5
    lisää_viiteen 7

    // Tämä on konseptina suurempi abstraktio kuin C# optionaaliset parametrit, ja mahdollistaa paremman laiskan evaluoinnin.
    // Sama plus voidaan ilmaista näin:
    let plus2 = (+) 
    // Miksi? Koska +-operaattori paitsi operaattori myös funktion ja sillä on tyyppi, jonka saa selville seuraavalla pätkällä koodi:
    (+).GetType()

module M5_Generics = 
    // Toisin kuin C#:ssa, F#:ssa kääntäjä generalisoi koodia automaattisesti.
    // Alla olevan funktion tyyppi on 'a -> string. 'a on funktio, joka ottaa geneerisen tyypin sisään ja palauttaa stringin ulos:
    let Method1 input = 
        input.ToString()
    //val Method2 : 'a -> string
    
    // Yleensä automatiikka toimii hyvin, joten manuaalinen eksplisiittinen tyypitys on usein turhaa.
    // Joskus silti kääntäjä voi tarvita pientä vinkkiä...

    // Tyypittää voi eksplisiittisesti:
    let Method2(input:int) :string = 
        input.ToString()
    //val Method1 : int -> string

    // Metodin paramterin voi määrittää myös ekplisiittisesti geneeriseksi
    let Method3(input:'t) =
        input.ToString()
    //val Method2 : 't -> string

    // F# ei ole käytännössä koskaan tarve vääntää C#:n tapaan rautalangasta, 
    // että 't on geneerinen tyyppi. Moinen on toki mahdollista:
    let Method4<'t>(input:'t) =
        input.ToString()

    // Kääntäjä tekee tarvittaessa inline-funktion kaikkiin missä kyseistä käytetään:
    // (Todella geneerinen mutta saattaa olla suorituskyvyltään huono, koska kääntäjä "kopioi" 
    // funktion kaikkialle missä sitä kutsutaan)
    let inline Method5 input = 
        input.ToString()
    Method5 "string"
    Method5 1

module M6_Generics_Vertailua = 
    // Lista-parametri OCaml-tapaan eksplisiittisesti: 
    let l2 : int list = [1;2;3]

    // Lista-parametri .NET-tapaan eksplisiittisesti:
    // (syntaksi eri, lopputulos käytännössä sama)
    let l1 : list<int> = [1;2;3]

    // Generics .NET-tapaan, tyypillä:
    type myType1<'t> = MyType1 of 't

    // Vastaava Generics OCaml-tapaan:
    // (syntaksi eri, lopputulos käytännössä sama)
    type 't myType2 = MyType2 of 't

    // Käyttö:
    let myThree1 = MyType1("something")
    let myThree2 = MyType2("something")

module M7_Rekursio = 
    // Rekursiivisen function esittelyyn pitää lisätä rec (pitkälti F# vahvan tyypityksen takia): 
    // Rekursiolla ei ole vaikutusta funktion tyyppiin (eli se ei vaihdu).
    // Huomaa, että funktion voi määritellä myös toisen funktion sisään. Rekursiivisten functioden osalta tämä on näppärä sääntö.
    let rec factorialPlus x =
        let rec factorialRec (x:int64) acc =
            if x > 1L then factorialRec (x - 1L) (x + acc)
            else acc
        factorialRec x 1L    

    factorialPlus 5L // 5 + 4 + 3 + 2 + 1 = 15
    factorialPlus 20000000L // val it : int64 = 200000010000000L

    /// Käytettäessä häntärekursiota F# hanskaa tilanteen jossa pino vuotaisi muuten yli (Stackoverflow). Häntärekursioksi kutsutaan rekursion erityistapausta, 
    /// jossa rekursiivisen kutsun paluuarvosta tulee ilman lisäoperaatioita kutsuvan instanssin paluuarvo. Tämä tarkoittaa sitä, että rekursioiden "purkautuessa" 
    // ei ole enää mitään tekemistä.
    let rec factorialPlusNonTail (x:int64) = 
        if(x > 1L) then
            // rekursiivisen kutsun jälkeen pitää lisätä sen palauttama 
            // arvo x:än ja sitten vasta palautetaan arvo. Koska yhteenlasku tapahtuu rekursiivisen 
            // kutsun palautettua arvon, kyseessä ei ole häntärekursio.
            x + (factorialPlusNonTail (x - 1L))
        else x
    factorialPlusNonTail  5L // 5 + 4 + 3 + 2 + 1 = 15
    factorialPlusNonTail  20000000L // Process is terminated due to StackOverflowException.

module M8_PatternMatching = 
    (*
    4. Pattern matching vaihtoehtona if-elselle 
     
     Lyhyesti: F#:n pattern matching on swich-case -rakenne tehtynä oikein
     Ks. http://msdn.microsoft.com/en-us/library/dd547125.aspx
    *)

    // Pattern matchingiä voi käyttää samaan tapaan kuin if-else sykliä voisi.     
    // Lopputulos on usein helppo lukuisempi ja tiiviimpi:
    //          |
    //     1    |     2
    //          |
    //  ---------------------
    //          |
    //     3    |     4
    //          |
    let resolveQuartile point =
        match point with
        | (x,y) when x >= 0 && y >= 0 -> 2
        | (x,y) when x < 0 && y >= 0 -> 1
        | (x,y) when x >= 0 && y < 0 -> 4
        | _ -> 3

    resolveQuartile (1,2)
    resolveQuartile (-1,2)
    resolveQuartile (1,-2)
    resolveQuartile (-1,-2)

    // F# Pattern matching tukee kymmenkuntaa eri "hahmontunnistus-kaavaa". Esimerkiksi parametrina annetun objektin tyyppi.
    // Esimerkki 2:
    let format (o: obj) =
        match o with
        | :? System.DateTime as day -> "Päivämäärä: " + day.ToString("dd.MM")
        | :? System.Int32 as i -> "Kokonaisluku: " + i.ToString()
        | _ -> "Jokin muu objekti"

    format (System.DateTime.Now)
    format 1
    format 1.0

    // Aktiiviset hahmoaihiot (active patterns): Haarautumisehto voidaan irrottaa kontekstistaan 
    // (Muistuttaa Clojuren multi-methods rakennetta):
    let (|Even|Odd|) input = if input % 2 = 0 then Even else Odd
    let TestNumber input = match input with Even -> printfn "%d is even" input | Odd -> printfn "%d is odd" input
    TestNumber 4
    TestNumber 7

module M9_TyypitJaOlioOrientoitunutOhjelmointi = 
    (*
    5. Tyypit 
    *)

    // F# on täysiverinen olio orientoitunut ohjelmointi kieli, mutta sen rakenteet kannustavat hyödyntämään muunlaisia 
    // rakenteita luokkia ja rajapintoja. Itse asiassa luokat ja rajapinnat eivät välttämättä ole edes paras mahdollinen 
    // lähtökohta uudelleenkäytettävälle ja elegantille olio-orientoituneelle koodille.
    // Usein tarkempi tekninen implementaatio (rajapinta/luokka/...) ei ole käyttäjälle merkityksellinen: olio-orientoitunut 
    // ongelman mallintaminen ei välttämättä auta löytämään hyvää ratkaisua ongelmaan.

    // Vertaa melun määrää:
    // C#-luokka:
    //    public class MyClass {
    //       public int Property { get ; private set; }
    //       public MyClass(int property){
    //            Property = property;
    //       }
    //    }

    // F#-luokka:
    type MyClass(property) =
      member x.Property = property

    // instanssin voi tehdä näin:
    let instance1 = MyClass ()
    // tai näin:
    let instance2 = new MyClass () // Huomasitko että F# tyypitti pametrin property generisesti 'unit'iksi.

    // type-avain sanaa voi käyttää myös aliaksen määrittämiseen:
    type dt = System.DateTime
    dt.Now
    
    // Rajapintoja hyödynnettäessä tosin kontravarianttia upcast-muunnosta eli tyyppimuunnosta kapeammasta tyyppistä 
    // yleisempään ei voi välttää. Kovarianttia tyyppimuunnosta ei juuri koskaan ole pakko käyttää eikä myöskään tulisi.
    // Tyyppi muunnosten syntaksi on seuraava:
    let myObject : obj = upcast "juttu"         // C#:ssa tämä ei ole tarpeen automaattisen kontravarianssin (string --> Object; 
                                                // erikoistuneempi --> yleisempi) takia. F# kontravarianssi ei ole automaattinen.
                                                // Näin ollen tämä muunnos vastaa erityisesti jos käyttää rajapintoja.
    let myString : string = downcast myObject   // Kovartiantti tyyppimuunnoss yleisemmästä --> erikoistuneenmpaan. Object --> String. 
                                                // Tätä suositallaan välttämään paitsi F#:ssa niin vahvasti tyypiteyissä kielissä 
                                                // ylipäänsä. Kovariantit tyyppimuunnokset tuottavat helposti ajonaikaisia bugeja. 
                                                // F#:ssa on tuskin koskaan tilanteita joissa downcast on usein järkevämpi vaihtoehto 
                                                // kuin hahmontunnistuksen (pattern matching) käyttö hyödyntäen tyypintunnistuskaavaa. 
    
    // ja sama operaattoreilla                                            
    let myObject3 = "juttu" :> obj // upcast eli kontravariantti tyyppimuunnos erikoistuneemmasta --> yleisempään.)
    let myString3 = myObject3 :?> string // downcast eli kovarianetti tyyppimuunnos yleisemmästä --> erikoistuneempaan.) 
    let integerType = typeof<int> // kuten C#:n typeof()

    // Tyyppimuunnoksille läheistä sukua oleva primitiivityyppien objektiin paketointi (boxing) ja objektista palauttaminen (unboxing) 
    // luonnistuu seuraavasti. Boxing on suhteellisen kallis operaatio, kun huomioi että yleensä sen voi välttää käyttämällä genericsiä. 
    let myStrObject, myIntAsObject = box("juttu"),box(int)
    let myString1, myInt1 = unbox<string>(myStrObject),unbox<int>(myIntAsObject)
    

module M10_TyypitJaOlioOrientoitunutOhjelmointi_Esimerkki = 
    // Alla oleva esimerkki havainnollistaa kuinka klassinen validointi dekoraattori on mahdollista toteuttaa 
    // olio-orientoituneesti käyttäen luokkia ja rajapintoja. Seuraavassa esimerkissä haviannollistetaan 
    // kuinka saman voi toteuttaa suunnilleen yhtä olio-orientoituneesti käyttämättä suoranaisesti 
    // kumpaakaan em. rakenteista.
    type IValidateInt = 
        abstract Validate: int -> bool 

    type LessThanValidator (max) =
        let max = max
        interface IValidateInt with 
            member x.Validate (intToValidate) = intToValidate < max
        
    type GreaterThanValidator (min) =
        let min = min
        interface IValidateInt with 
            member x.Validate (intToValidate) = intToValidate > min
    //...

    let lessThan10Validator = new LessThanValidator(10) :> IValidateInt
    let moreThan5Validator  = new GreaterThanValidator(5) :> IValidateInt
    lessThan10Validator.Validate 9 // true
    lessThan10Validator.Validate 12 // false
    moreThan5Validator.Validate 2 // false
    moreThan5Validator.Validate 8 // true

    // Tätä voisi jatkaa ylikuormittamalla && ja || operaattorit tuottamaan AndValidator-luokan, jne.
    // Mutta koodin määrä paisuu yli simppelin esimerkin. 

    // Voisi tehdä myös anonyymilla luokalla, jos instansseja on vain yksi. 
    // Objekti ilmaukset (Object Expression) on C#:n anonyymien tyyppien "parempi versio". C# kääntäjä ei salli rajapinnan 
    // IValidateInt toteuttavaa anonyymiä tyyppiä; F# sallii: 
    let lessThan7Validator = { new IValidateInt with member x.Validate(i) = i<7 }
    lessThan7Validator.Validate 6   // true
    lessThan7Validator.Validate 8   // flase

    // Rajanpinnan käyttö tekee rakenteesta asteen joustavamman, mutta koodia pahaisen ominaisuuden määrittelyyn 
    // tarvitsee kirjoittaa enemmän kuin itse sovellus logiikkan monimutkaisuus soisi. 

    // F#:n jäsennelty unioni (discriminate union) tuo ratkaisun tähän. Oheinen ratkaisu on huomattavasti
    // monipuolisempi mutta ei juurikaan monimutkaisempi tai pidempi. Tarkasti ottaen jäsennelty unioni 
    // kääntyy abstraktiksi luokaksi, jolla on sen itsensä periviä sisäluokkia. Perinteisesti ymmärrettynä
    // se ei ole luokka vaan jotain muuta. (Vähän kuin "joko-tai-luokka".)
    
    type ValidateInt =
        | GreaterThan of int
        | LessThan of int
        | Predicate of (int -> bool)
        | And of ValidateInt * ValidateInt
        | Or of ValidateInt * ValidateInt
        member x.Validate intToValidate =
            match x with
            | GreaterThan max -> intToValidate > max
            | LessThan min -> intToValidate < min
            | Predicate validationFunction -> validationFunction intToValidate
            | And (validator1, validator2) -> (validator1.Validate intToValidate) && (validator2.Validate intToValidate)
            | Or (validator1, validator2) -> (validator1.Validate intToValidate) || (validator2.Validate intToValidate)

    let lessThan20Validator = LessThan  20 
    let moreThan20Validator  = GreaterThan  20 
    lessThan20Validator.Validate 9 // true
    lessThan20Validator.Validate 22 // false
    moreThan20Validator.Validate 9 // false
    moreThan20Validator.Validate 22 // true

    // Ja sitten jotain ihan muuta. Validoidaan onko int joko (1) parillinen ja yli 10 tai (2) pariton ja alle kymmenen
    let complexValidator1 = 
        let isOddValidator = ValidateInt.Predicate (fun x -> (x % 2) = 1)
        let isEvenValidator = ValidateInt.Predicate (fun x -> (x % 2) = 0)
        Or (And ((GreaterThan 10), isEvenValidator), (And ((LessThan 10), isOddValidator)))  
    complexValidator1.Validate 9 // true
    complexValidator1.Validate 13 // false
    complexValidator1.Validate 8 // false
    complexValidator1.Validate 12 // true

    // Ratkaisu toimii mutta syntaksin edellyttämä sulkuhässäkästä on erittäin vaikea lukuinen.
    // F# sallii luettavuutta helpottavien prefix ja infiksi operaattorien luonnin. Tehdään tässä niin:
    //
    // Harmittavasti CLI (Common Language Infrastructure) ei anna määrittää && tai || operaattoreja 
    // jotka palauttavat jotain muuta kuin booleanin.  (op_BooleanAnd toimisi, mutta return-tyyppi on väärä)
    // Käytössä ovat esim.  &&& ja |||  tai  +& ja +|  mutta molemmat ovat hieman kryptisiä nimiä, mutta 
    // parempia kuin ei mitään.
    //
    // Uusia operaattoreja voi määrittää osana luokan esittelyä tai globaalisti sen esittelyn jälkeen näin:
    let inline (&&&) (fst:ValidateInt) (snd : ValidateInt) =
            ValidateInt.And (fst,snd)
    let inline (|||) (fst:ValidateInt) (snd:ValidateInt) =
            ValidateInt.Or (fst,snd)

    let complexValidator2 = 
        let isOddValidator = ValidateInt.Predicate (fun x -> (x % 2) = 1)
        let isEvenValidator = ValidateInt.Predicate (fun x -> (x % 2) = 0)
        (GreaterThan 10 &&& isEvenValidator) ||| (LessThan 10 &&& isOddValidator)
    complexValidator2.Validate 9 // true
    complexValidator2.Validate 13 // false
    complexValidator2.Validate 8 // false
    complexValidator2.Validate 12 // true

    // F# luokkia on mahdollista laajentaa lennosta with-avainsanalla. 
    //
    // Rakenne muistuttaa kaukaisesti C#:n laajennus metodeja (extension methods) tai mahdollisuutta esitellä luokka useassa 
    // eri tiedostossa käyttäen partial määrettä luokassa. Koodi on edelleen vahvasti tyypitettyä, sillä tätä tyyppilaajennusta 
    // ei hyödynnetä tätä ennen.
    type ValidateInt with
        static member Any (alternatives:ValidateInt list) = 
            alternatives 
            |> List.fold 
                (fun (state : ValidateInt option) (current:ValidateInt) -> 
                    match state with 
                    | Some acc -> Some (current ||| acc) 
                    | None -> Some current) None
            |> fun this -> match this with Some v -> v | None  -> Predicate (fun i -> true)
        member x.OrAny (alternatives:ValidateInt list) = ValidateInt.Any (x :: alternatives)            
    
    // Muokataan edellisen kohdan validaattoria seuraavan laiseksi
    // 1) 23 on validi aina
    // 2) Parilliset luvut ovat valideja, jos ne ovat suurempia kuin kymmenen
    // 3) Parittomat luvut ovat valideja, jos ne ovat pienempi kuin kymmenen (poislukien tietty luvun 23)
    let complexValidator3 =
        let isOddValidator = ValidateInt.Predicate (fun x -> (x % 2) = 1)
        let isEvenValidator = ValidateInt.Predicate (fun x -> (x % 2) = 0)
        let is23Validator = ValidateInt.Predicate (fun x -> x = 23)
        ValidateInt.Any [is23Validator; (GreaterThan 10 &&& isEvenValidator);  (LessThan 10 &&& isOddValidator)]
    complexValidator3.Validate 9 // true
    complexValidator3.Validate 13 // false
    complexValidator3.Validate 23 // true
    complexValidator3.Validate 8 // false
    complexValidator3.Validate 12 // true
      
module M11_LoopitJaListaOperaatiot = 
    (*
    6. Loopit ja listaoperaatiot
    *)

    // Klassinen for-loopi toimii näin:
    for i = 1 to 3 do printfn "Jee %d" i
    for i = 3 downto 1 do printfn "Jee %d" i

    // C#:n foreachia vastaava rakenne näyttää F#:ssa tältä.
    for i in [1;2;3] do printfn "Jee %d" i

    // Usein for loopeja tehokkaampaa ja elegantimpaa on putkittaa komentoja. |> operaattorilla
    // (Yksinkertaisissa tapauksissa putkitus ei tosin selvennä juurikaan koodia.)
    let anotherList = [1;2;3]
    anotherList |> List.iter (printfn "Jee %d") 

    // Yksittäiseen alkioon viittaaminen tapahtuu seuraavasti:
    let simpleList = [1;2;3;4] 
    let fourth = simpleList.[3]

    // List.iter on epäilemättä yksi tavanomaisemmin hyödynnettävistä F#:n operaattoreista. Muita kiinnostavia funktioita ovat mm. 
    // map, fold, filter, sortBy, forall ja exists.
    //
    // Seuraavaa koodi esimerkki havainnollistaa monien eri lista operaattorien hyödyntämisestä yhdessä toisiinsa putkitettuna työnkulkuna:
    // 1) filter tuottaa listan jossa on vain filteröinti ehdon täyttävät rivit. Se vastaa LINQ:n Whereä
    [1;2;3] |> List.filter (fun i -> i % 2 = 1) // palauttaa [1;3]
    // 2) forall palauttaa true:n jos ja vain jos kaikkille riveille jokin annettu ehto on tosi. forall vastaa LINQ:n All:ia.
    [1;2;3] |> List.forall (fun i -> i % 2 = 1) // false; kaikki eivät ole parittomia
    [1;2;3] |> List.exists (fun i -> i % 2 = 1) // true; on vähintään yksi pariton
    // 3) sortBy on vastaa LINQ:n SortBy:ta ja nimensämukaseisti järjestää elementit nousevaan suuruusjärjestykseen - siten että korkein 
    // arvo on ensin. Päin vastaisen järjestyksen saa aikaiseksi helposti esim hyödyntämällä vastalukua tai hyödyntämällä listan kääntävä
    // rev:iä. Sort on sortBy objektilla joka on vertailtavissa (comparable) itsessään ja sellaisenaan.
    [2;3;1] |> List.sort                 // [1; 2; 3]
    [2;3;1] |> List.sortBy (fun i -> -i) // [3; 2; 1]
    [2;3;1] |> List.sort |> List.rev     // [3; 2; 1]

    // 4) fold vastaa LINQ:n Aggregatea. Aggregate riviriviltä päivittää eli tuottaa uuden arvonkerryttimen (accumulator). 
    // Esim. listan summa on aggregaatti jossa ensin nollaan on lisätty ensimmäisen rivin arvo, sitten ensimmäisen rivin 
    // arvoon toisen rivin arvo, ja sitten kolmanteen ensimmäisen ja toisen rivin summa jne. kunnes N:nteen riviin lisätään 
    // kaikkea sitä edeltävien rivien akkumuloitu summa. Alla olevat lyhyt rivi selventää foldin käyttöä
    [1;2;3] |> List.fold (+) 0 // palautta: (((0+1)+2)+3) = 6. Sulkeet lisätty selventämään mitä missäkin loopin syklissä tapahtuu. 
    // 5) map ja LINQ:n Select ovat molemmat projektioita listasta tyyppiä list 'T toiseen listaan tyyppiä list 'u
    [1;2;3] |> List.map (fun x->x+1) //[2; 3; 4]

    // Alla on myös havainnollistettu tietuetyypin (record type) käyttöä sekä sitä kuinka funktioita voi käyttää ilman 
    // sen suurempia kommervenkäjä listan yksittäisinä kohteina. 
    
    // Record type muistuttaa C# structia kuitenkaan olematta struct. Sen keskeisin käyttö tarkoitus on toimia oletuksena 
    // ei-muutettavana ja vahvasti tyypitettynä säilönä kompleksiselle datalle.
    type Candidate =
        {
            Name : string                           
            Age : int
            WorkExperience: float
            HighschoolGeneralGrade: float
        }
    let requirementRules = // Huomaa: jos listan yksittäisen rivin esittelee aina omalla rivillään 
        [                  // ilmauksen lopettavaa ";":ttä ei tarvitse käyttää!
        (fun c -> c.Age >= 18)
        (fun c -> c.HighschoolGeneralGrade >= 7.0)
        (fun c -> c.WorkExperience >= 0.5)]
    let scoringRules = 
        [
        (fun c -> c.HighschoolGeneralGrade / 9.0)
        (fun c -> match c.WorkExperience  with | exp when exp < 2.0 -> 1.0 | exp when 2.0 <= exp && exp < 10.0 -> 1.1 | _ -> 1.3   )
        ]
    let candidates = 
        [   // F# arvaa varsin usein mitä tietuetyyppiä käyttäjä haluaa käyttää sen sisältämien kenttien nimistä.
            // Jos näin ei käy voi ensimmäisen kentän eteen laittaa tyypin nimen auttamaan kääntäjää seuraavaan tapaan:
            {Candidate.Name="Tim"; Age=30; WorkExperience=4.0; HighschoolGeneralGrade=8.2}
            // Tyypillisesti seuraava riittää:
            {Name="Anna"; Age=17; WorkExperience=0.0; HighschoolGeneralGrade=9.2}
            {Name="Elsa"; Age=37; WorkExperience=15.0; HighschoolGeneralGrade=7.2}
            {Name="Arnold"; Age=42; WorkExperience=22.0; HighschoolGeneralGrade=5.2}
            {Name="Emily"; Age=27; WorkExperience=2.0; HighschoolGeneralGrade=9.8}
            {Name="Erwin"; Age=22; WorkExperience=1.0; HighschoolGeneralGrade=8.8}
        ]
    let ranking = 
        candidates
        // Ensin suodatetaan joukosta pois ne, jotka eivät täytä minini vaatimuksia
        |> List.filter (fun c -> requirementRules |> List.forall (fun f -> f c))
        // Sitten lasketaan rankking arvo kertomalla 1:llä kaikkin ränkkin funktioiden antamat tulokset
        |> List.sortBy (fun c -> scoringRules |> List.fold (fun (acc:float) (f) -> acc * f c) 1.0)
        // Lopuksi valitaan näytettäväksi nimi, käyttäen map-funktiota
        |> List.map (fun c -> c.Name)
        // Lopuksi käännetä järjestys ympäri, koska oletuksena järjetys on pienemmästä suurempaan
        |> List.rev

    // Lisää jokaiseen lista elementtiin kaikien sitä edeltäneiden elementtien summa:
    let increaseBySumOfPrecessors list = 
        let initialAccumulator = (0, List.empty<int>)
        let processItem ((sumAccumulator:int), (listAccumulator:int list)) (current:int) =
            let newSum = sumAccumulator + current
            let newList = newSum::listAccumulator 
            newSum, newList 
        let returnList ((sumAccumulator:int), (listAccumulator:int list)) = 
            listAccumulator
            |> List.rev // accumulator add new item alway to the beginning of list - result list is reverse version of that.
        list |> List.fold processItem initialAccumulator |> returnList
    increaseBySumOfPrecessors [1;2;3;5;7;10]
    
    // Sequence expression: For looppia voi käyttää listojen generointiin. 
    open System // Yleisesti ottaen selkeintä on avata tarvittavat nimiavaruudet modulen alussa. Skripteissä nimiavaruuden 
                // voi avata periaatteessa milloin vain. Käännetyssä koodissa paikka on suuremmassa määrin rajoitettu.
    let firstday2012 = new DateTime(2012,1,1)
    let year2012 = seq {for i in 0.0 .. 365.0 -> firstday2012.AddDays(i)} // Tässä luodaan sekvenssi erityistä seq builder syntaksia hyödyntäen
                                                                          // tämä tapa esitellä lista on erityisen näppärä kun käsitellään
                                                                          // äärettömän pitkiä listoja. Esimerkki tästä alla. 

    // Sekvenssi voidaan tehdä myös yield-käskyllä, joka on sama kuin C#:ssa. (Visual basiciin yield tulili vasta versioon 9, PowerShellistä se
    // se puuttuu edelleen, mikä tosin ei kielen luonteen takia juurikaan haittaa. 
    // F# tukee seq-operaatioissa myös avain sanaa yield! joka yieldaa koko setin osaksi paluuarvoa:
    let rec iterate f value = 
      seq { yield value; 
            yield! iterate f (f value) }
    
    let xs = iterate (fun f-> f + 1) 0
             |> Seq.take(10) |> Seq.toList

    let nextThenHours = 
        iterate (fun (f : System.DateTime) -> (f.AddHours(1.0))) System.DateTime.Now 
        |> Seq.take(10) 
        |> Seq.toList
    
    
    // Kun loopin logiikka monimutkaistuu, komentojen putkitus alkaa selkeyttämään koodia enenevässä määrin.
    // Funktionaalinen ohjelmointiparadigma on vahvimmillaan nimenomaan monimutkaisten ongelmien parissa puuhatessa.
    year2012 
    |> Seq.filter (fun day -> day.DayOfWeek = DayOfWeek.Friday && day.Day = 13)
    |> Seq.iter (printfn "%O")

    // Operaatiokirjasto on vähän monipuolisempi kuin LINQ, joten sillä pääsee pitkälle.
    // Kun voima loppuu kesken, astuu kuviin rekursio ja pattern matching.

    // Seuraavassa esimerkissä näytetään tämän vuoden perjantai 13. -päivät ja milloin vastaava päivämäärä on perjantai seuraavan kerran.
    let rec findNextSameFriday13 (day : DateTime) =
        let next = new DateTime(day.Year + 1, day.Month, day.Day)
        match next.DayOfWeek with
        | DayOfWeek.Friday -> next
        | _ -> findNextSameFriday13 next
    
    year2012 
    |> Seq.filter (fun day -> day.DayOfWeek = DayOfWeek.Friday && day.Day = 13)  //Suodatus: filter = "where/reduce/..."
    |> Seq.map (fun day -> (day, findNextSameFriday13 day))  //Mappaus tyypistä toiseen: "projektio/select/..."
    |> Seq.iter(fun (day2012, dayN) ->  (printfn "%s on seuraavan kerran perjantai 13. vuonna %d" (day2012.ToString("dd.MM")) dayN.Year))

    // F# sallii äärettömät sekvenssit. Seuraavaksi etsitään ensimmäisen perjantaina joka on kolmastoista päivä ja jolle 
    // vuosi * kuukausi * päivä on suurempi kuin 1 000 000. 
    // Esim. jos 13.12.2012 olisi perjantai vuoden, kuukauden ja päivän tulo (13 x 12 x 2012) olisi 313872. Tässä toinen esimerkki
    // äärettömistä listoista ja siitä mitä hyötyä sellaisesta on. 

    // 1. Ensin koodia selventävä apufunktio seuraavan kolmannentoista päivän hakemiseen. 
    let rec findNextFriday (day : DateTime) =
        let next = day.AddDays(1.0)
        match (next.Day, next.DayOfWeek) with
        | 13, DayOfWeek.Friday -> next
        | _ -> findNextFriday next

    // 2. Tämän jälkeen luodaan ääretön sekvenssi perjantai kolmastoista päiviä
    let fri13seq = Seq.unfold (fun state -> Some(state, (findNextFriday state))) (new DateTime(2012,1,13))
    // 3. Lopuksi iteroidaan sekvenssiä läpi kunnes ehdot täyttävä päivä löytyy.
    fri13seq |> Seq.skipWhile (fun day -> day.Year * day.Month * day.Day < 1000000) |> Seq.head

    // F# lista on oletuksena linkitetty lista, jonka ensimmäiseksi alkioksi lisääminen on tehokas operaatio
    // (koska lista on muuttumaton ("immutable"), niin uuden alkion lisäys eteen on vain uusi alkio ja pointteri vanhaan listaan)
    let emptyList = []
    let listOfListOfIntegers = [[1;2;3];[4;5;6]]

    // Listoja voi yhdistellä (merge):
    let merged1to6 = [1;2;3] @ [4;5;6]
    //val merged1to6 : int list = [1; 2; 3; 4; 5; 6]

    // lisäksi usein käsitellään ensimmäistä alkiota, ja välitetään loput rekursiolle. :: erottaa ensimmäisen alkion ja loput:
    let mylist = "head" :: ["tail1"; "tail2"]
    mylist
    // Tyypillinen match koontifunktiolle on jotain tämän suuntaista:
    // (Yksinkertaistettuna, tämänhän voi tehdä vielä myös peruslistaoperaatioilla)
    let rec sample f x =
        match x with
        | [] -> 0
        | h::t -> f(h) + sample f t 

    // Esim. Fibonacci-lukusarja:
    let rec fibs a b = 
        match a + b with c when c < 10000 -> c :: fibs b c | _ -> [] 
    let fibonacci = 0::1::(fibs 0 1) 

    // Matriisit ja moniulotteiset arrayt:
    Array2D.init 3 3 (fun x y -> x+y)
    |> Array2D.map (fun a -> a+1)
    // val it : int [,] = [[1; 2; 3]
    //                     [2; 3; 4]
    //                     [3; 4; 5]]
    