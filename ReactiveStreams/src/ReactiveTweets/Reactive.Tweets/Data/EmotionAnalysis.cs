using edu.stanford.nlp.ling;
using edu.stanford.nlp.neural.rnn;
using edu.stanford.nlp.pipeline;
using edu.stanford.nlp.sentiment;
using edu.stanford.nlp.trees;
using java.util;
using System;
using System.Linq;
using Tweetinvi.Models;

namespace Reactive.Tweets
{
    public class EmotionAnalysis
    {
        public EmotionAnalysis(string modelsFolder)
        {
            var properties = new Properties();
            properties.setProperty("annotators",
                "tokenize,ssplit,pos,parse,sentiment");

            var temp = Environment.CurrentDirectory;
            Environment.CurrentDirectory = modelsFolder;
            stanfordNLP = new StanfordCoreNLP(properties);
            Environment.CurrentDirectory = temp;

            emotionAnnotationTreeClassName =
                new SentimentCoreAnnotations.SentimentAnnotatedTree().getClass();
            sentencesAnnotationClassName =
                new CoreAnnotations.SentencesAnnotation().getClass();
        }

        private readonly StanfordCoreNLP stanfordNLP;
        private readonly java.lang.Class emotionAnnotationTreeClassName;
        private readonly java.lang.Class sentencesAnnotationClassName;

        public class Emotion
        {
            public Emotion(int num)
            {
                _num = num;
            }
            private readonly int _num;
            public override string ToString()
            {
                switch (_num)
                {
                    case 0: case 1:
                        return "Unhappy";
                    case 2:
                        return "Indifferent";
                    case 3: case 4:
                        return "Happy";
                    default:
                        throw new ArgumentException($"Unknown emotion value '{_num}'");
                }
            }
        }

        public Emotion GetEmotion(string text)
        {
            var annotation = new Annotation(text);
            stanfordNLP.annotate(annotation);

            return
                (annotation.get(sentencesAnnotationClassName) as ArrayList)
                .toArray().Select(ParseEmotion).FirstOrDefault();

            Emotion ParseEmotion(object s) {
                var sentence = s as Annotation;
                var sentenceTree = sentence.get(emotionAnnotationTreeClassName) as Tree;
                var emotion = RNNCoreAnnotations.getPredictedClass(sentenceTree);
                return new Emotion(emotion);
            }
        }

        public (ITweet, Emotion) AddEmotion(ITweet tweet)
        {
            var emotion = GetEmotion(tweet.Text);
            return (tweet, emotion);
        }
    }
}
