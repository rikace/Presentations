(*
Verkoston noodien värjäys ongelmassa ideana on värjätä jokainen graafin noodi siten,
että yksikään sen vierusnoodeista ei ole saman värinen. Varsin tavanomainen 
kartografinen variaatio yleisestä väräysongelmasta kuuluu seuraavasti: Miten Etelä-Amerikan 
maat tulisi värjätä, siten että yhdenkään maan naapuri ei ole saman värinen? 

Eric Lippert (C#-kääntäjän kehitystiimistä) käsittelee blogissaan ongelmaa perusteellisesti viiden 
kirjoituksen verran:
http://blogs.msdn.com/b/ericlippert/archive/2010/07/12/graph-colouring-with-simple-backtracking-part-one.aspx
http://blogs.msdn.com/b/ericlippert/archive/2010/07/15/graph-colouring-with-simple-backtracking-part-two.aspx
http://blogs.msdn.com/b/ericlippert/archive/2010/07/22/graph-colouring-with-simple-backtracking-part-three.aspx
http://blogs.msdn.com/b/ericlippert/archive/2010/07/26/graph-colouring-part-four.aspx
http://blogs.msdn.com/b/ericlippert/archive/2010/07/29/graph-colouring-part-five.aspx

Jon Harper on kääntänyt blogissaan ongelman F#:ksi http://fsharpnews.blogspot.com/2012/05/graph-coloring-in-f.html

Harmittavasti Jon Harperin ratkaisu on kuitenkin varsin hankala seurata ja ymmärtää. 
Alla oleva ratkaisu ongelmaan mukailee Jon Harperin ratkaisua mutta on toivoakseni merkittävästi 
helpompi ymmärtää ja seurata.

Harperin pointti oli osoittaa että F# on hemmetin tehokas kieli tällaisiin C# ratkaisussa ongelma domainin määrittelyyn 
käytetiin 20 riviä koodia ja itse ratkaisu vaati noin 200 riviä C# koodi (Harperin mukaan; en ole tarkastanut tätä). 
Harperin ratkaisi ongelman 19 rivillä ja niin tehdessään joutui tinkimään luvattoman paljon luettavuudesta ja selkeydestä. 
Alla oleva huolellisesti kommentoidulla ratkaisulla on pituutta reilu 60 riviä kommentit mukaan lukien - eli edelleen lähes 4 kertaa 
vähemmän kuin C# verrokissa.

Koodi rivien määrällä sinänsä ei ole väliä - kokonaisuus ratkaisee. Useimmissa tapauksissa 200 sivuisen kirjan lukee ja 
sisäistää nopeammin kuin 800-sivuisen ihan vaan siitä yksinkertaisesta syystä että sisäistettävää ja luettavaa on vähemmän. 
Sama pätee koodiin: 50 riviä nyt vaan on nopeampi sisäistää kuin 200 ja 200 riviä nopeampi kuin 800. 

Toki, jos kirja sattuu olemaan Saksalaista 1900-luvun vaihteen filosofiaa jossa on säästelty kylläkin sivuissa muttaei sivun 
mittaisissa lauseissa, lienee ilmi selvää, että pienen sivumäärän lukemiseen ja sisäistämiseen herkästi kymmeniä kertojen 
pidempään kuin helppolukuisen. Sama pätee koodiin: kryptisiin äärimmäisen tiiveiksi puristettuihin säännöllisiin lausekkeisiin
menee tuhottomasti aikaa verrattuna mihin tahansa muuhun koodiin. 

Onneksi tiiviys ei implikoi kryptisyyttä sen enempää kirjoitettujen tekstien kuin koodinkaan osalta: Tiivisti voi 
kirjoittaa olematta kryptinen ja jos siinä onnistuu teksti on todennäköisesti mitä erinomaisinta. Vastaavalla tavalla 
tiivis koodi voi olla hyvinkin helppolukuista. Usein juurikin koodin määrä tekee siitä hankalammin hahmotettavaa. 

Asiaa voisi ajatella näinkin: Miksi ylipäätään käytätte for-looppia copy&pasten sijaan?
Koodin toiminnallisuuden läpi miettiminen kestää aikansa, oli koodi kuinka lyhyt tai pitkä hyvänsä.
Lyhyempi koodi, korkeampi abstraktio, tarkempi kääntäjä ja pienempi manuaalinen virheherkkyys johtavat jo itsessään
koodiin jonka ylläpitokustannukset ovat pienemmät.

*)
module Data = 
    type Country =
        | Brazil | FrenchGuiana | Suriname | Guyana | Venezuala | Colombia
        | Ecuador | Peru | Chile | Bolivia | Paraguay | Uruguay | Argentina
 
    let countries =
        [|Brazil; FrenchGuiana; Suriname; Guyana; Venezuala; Colombia; Ecuador; Peru; Chile; Bolivia; Paraguay; Uruguay; Argentina|]
 
    let edges =
        [
            Brazil, [ FrenchGuiana; Suriname; Guyana; Venezuala; Colombia; Peru; Bolivia; Paraguay; Uruguay; Argentina ]
            FrenchGuiana, [ Brazil; Suriname ]
            Suriname, [ Brazil; FrenchGuiana; Guyana ]
            Guyana, [ Brazil; Suriname; Venezuala ]
            Venezuala, [ Brazil; Guyana; Colombia ]
            Colombia, [ Brazil; Venezuala; Peru; Ecuador ]
            Ecuador, [ Colombia; Peru ]
            Peru, [ Brazil; Colombia; Ecuador; Bolivia; Chile ]
            Chile, [ Peru; Bolivia; Argentina ]
            Bolivia, [ Chile; Peru; Brazil; Paraguay; Argentina ]
            Paraguay, [ Bolivia; Brazil; Argentina ]
            Argentina, [ Chile; Bolivia; Paraguay; Brazil; Uruguay ]
            Uruguay, [ Brazil; Argentina ]
        ]

module Solver = 
    open Data
    let allColors = set [0..3]

    let solve (countyVerticles:Country array, countryNeighborMapping: (Country * Country list) list) =
        // Tätä apufunktio käytetään tarkastamaan voisiko tällä hetkellä käsittelyssä olevan maan värjätä 
        // värillä colorToCompare.
        let haveAllDifferentColorThan (colorMap: Map<Country,int>) (neighbors:Country seq) colorToCompare  = 
            let hasDifferentColor (country:Country) = colorMap.[country] <> colorToCompare
            neighbors |> Seq.forall hasDifferentColor

        let foldSolutionFunc (processedCountries, solutions) countryInProcess =
            // Etsitään kaikki ne naapurit, jotka on jo käsitelty ja 
            // tarkatetaan onko maa itse mahdollisesti jo käsitelty osittain osana
            // jonkin toisen maan käsittelyä.
            let processedRelatedCountries =
                [|for country, neighbors in countryNeighborMapping do
                    if country = countryInProcess then 
                        yield! Set.intersect processedCountries (set neighbors) 
                    else
                        for neighbor in neighbors do
                            if neighbor = countryInProcess && Set.contains country processedCountries then
                                yield country|]
            Set.add countryInProcess processedCountries,
            // Muodostetaan lista kaikista mahdollisista maavärikartoista.
            //
            // Ensimmäisellä kerralla sellaista ei ole tällöin ensimmäinen maa saa 
            // ensimmäisen värin, toinen toisen kolmas kolmannen ja neljäs neljännen
            // jos maita sattui olemaan enemmän, niin ne käsitellään myöhemmillä iteraatioilla
            // 
            // Seuraavalla maalle tehdään samoin sillä erolla, että aina tarkastetaan onko jollekin sen naapurimaalle jo 
            // annettu sama väri. Väri lisätään listaan vain jos listassa ei ole Kun kaikki maat on käyty tällä tavalla.
            //
            // Viimeisenä jonosta pullahtaa ulos sellainen ratkaisu, jossa kaikille maille on annettu jokin väri. Toimivia 
            // ratkaisuja voi toki olla useita ja viimeinen on varmuudella jokin niistä, jos ratkaisu ylipäänsä löytyi. Ellei löytynyt 
            // niin ratkaisu listasta puuttuu jokin maa.
            seq { for possibleColorMap in solutions do
                    for newColor in allColors do
                        if haveAllDifferentColorThan possibleColorMap processedRelatedCountries newColor then
                            yield Map.add countryInProcess newColor possibleColorMap }
        let accumulator = (Set.empty, seq[Map.empty])
        countyVerticles |> Seq.fold foldSolutionFunc accumulator

let answer =
    let _, colors = Solver.solve (Data.countries, Data.edges)
    // Viimeisenä ulos pullahtanut ratkaisu sisältää kaikki maat (jos tähän annettuu ongelmaan 
    // ylipäänsä on ratkaisua.)
    let color = Seq.head colors
    // Lopuksi mäpätään ratkaisu maihin.
    [ for country in Data.countries -> country, color.[country] ]