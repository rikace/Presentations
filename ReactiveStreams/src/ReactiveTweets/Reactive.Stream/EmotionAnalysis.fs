namespace Reactive.Emotion

type EmotionType = { emotion:int }
                   override this.ToString() =
                      match this.emotion with
                      | 0 | 1 -> "Unhappy"
                      | 2 -> "Indifferent"
                      | 3 | 4 -> "Happy"
                      | x -> failwith (sprintf "Unknown emotion value %d" x)

module Analysis =

    open edu.stanford.nlp.ling
    open edu.stanford.nlp.neural.rnn
    open edu.stanford.nlp.pipeline
    open edu.stanford.nlp.sentiment
    open edu.stanford.nlp.trees
    open java.util
    open System
    open System.Linq
    open Tweetinvi.Models

    let modelsFolder = "../../../models"
    let properties = Properties()
    properties.setProperty("annotators", "tokenize,ssplit,pos,parse,sentiment") |> ignore

    let temp = Environment.CurrentDirectory
    Environment.CurrentDirectory <- modelsFolder
    let stanfordNLP = new StanfordCoreNLP(properties)
    Environment.CurrentDirectory <- temp

    let emotionAnnotationTreeClassName = 
        let sTree = new SentimentCoreAnnotations.SentimentAnnotatedTree()
        sTree.getClass()

    let sentencesAnnotationClassName = 
        let sentence = new CoreAnnotations.SentencesAnnotation()
        sentence.getClass()

    let parseEmotion (o:obj) =
        let sentence = o :?> Annotation
        let sentenceTree = sentence.get(emotionAnnotationTreeClassName) :?> Tree
        let emotion = RNNCoreAnnotations.getPredictedClass(sentenceTree)
        { emotion = emotion } 

    let getEmotion (text:string) =
        let annotation = new Annotation(text)
        stanfordNLP.annotate(annotation)
        let arr = annotation.get(sentencesAnnotationClassName) :?> ArrayList
        arr.toArray()
        |> Array.map(fun o -> parseEmotion o)
        |> Array.head
    
    let addEmotion (tweet:ITweet) =
        let emotion = getEmotion tweet.Text
        (tweet, emotion)
