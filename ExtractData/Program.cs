
using Catalyst;
using Catalyst.Models;
using edu.stanford.nlp.ie.crf;
using edu.stanford.nlp.ling;
using java.util;
using Mosaik.Core;
using NReco.NLQuery;
using NReco.NLQuery.Table;
using System.Text;

namespace ExtractData
{
    internal class Program
    {
        static void Main(string[] args)
        {
            StanfordNLP();

            //await UsingCatalyst();

            //NLQuery();
        }

        //gets pronouns
        private static async Task UsingCatalyst()
        {
            //get text input
            string text = GetTextInput();
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading model...");

            //english model is registeration
            English.Register();

            ////storage folder for downloaded models(optional)
            //Storage.Current = new DiskStorage("catalyst-models");

            //create an NLP pipeline for English
            var nlp = await Pipeline.ForAsync(Language.English, sentenceDetector: true, tagger: true);

            //create and process a document
            var doc = new Catalyst.Document(text, Language.English);
            nlp.ProcessSingle(doc);

            //get results from Span(PROPNs)
            Console.ResetColor();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Green;
            foreach (var span in doc.Spans)
            {
                foreach (var token in span.Tokens)
                {
                    if (token.POS == PartOfSpeech.PROPN)
                    {
                        Console.WriteLine($"{token.POS} : {token.Value}");
                    }
                }
            }
            Console.ResetColor();
            Console.ReadLine();
        }

        //no accurate results
        private static void NLQuery()
        {
            //build a matcher builder with schema
            var tblMatchBuilder = new TableMatcherBuilder();
            // tblMatchBuilder.Add(yourTableSchema);

            //add match builder to recognizer
            var recognizer = new Recognizer(tblMatchBuilder.Build());

            //get text input
            string userInput = GetTextInput();
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading model...");

            // Tokenize input
            var tokens = new NReco.NLQuery.Tokenizer().Parse(userInput);
            var tokenSeq = new TokenSequence([.. tokens]);

            //recognize entities
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Green;
            recognizer.Recognize(tokenSeq, matches =>
            {
                Console.WriteLine($"Matched token/entity: {matches}");
                //Console.WriteLine($"  Hint: {match.Hint}, Value: {match.Value}");
                foreach (var m in matches)
                    Console.WriteLine($"{m}");

                return true;
            });
            Console.ResetColor();
            Console.ReadLine();

        }

        //extracts names, organization, locations, date
        //https://github.com/wolfgangmm/exist-stanford-ner/blob/master/resources/classifiers
        //https://www.youtube.com/watch?v=JIz-hiRrZ2g
        static void StanfordNLP()
        {
            //get text input
            string text = GetTextInput();
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading model...");

            string modelPath = "ExtractData.TrainingModels.english.all.3class.distsim.crf.ser.gz";

            // Load classifier
            var classifier = CRFClassifier.getClassifierNoExceptions(modelPath);

            // Classify text
            var sentences = classifier.classify(text);

            List<string> personNames = [];
            foreach (List sentence in sentences.toArray())
            {
                StringBuilder name = new();

                foreach (var obj in sentence.toArray())
                {
                    var token = (CoreLabel)obj;
                    string word = token.word();
                    string tag = token.get(typeof(CoreAnnotations.AnswerAnnotation)).ToString();
                    //string tag1 = token.get(typeof(CoreAnnotations.CalendarAnnotation))?.ToString();
                    //Console.WriteLine($"{tag} - {word}");

                    if (tag == "PERSON")
                    {
                        if (name.Length > 0) name.Append(' ');
                        name.Append(token);
                    }
                    else
                    {
                        if (name.Length > 0)
                        {
                            personNames.Add(name.ToString().Trim());
                            name.Clear();
                        }
                    }
                }
            }

            Console.ResetColor();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Green;
            var distinctNames = personNames.Distinct().ToList();
            Console.WriteLine("Extracted human names: ");
            Console.WriteLine(string.Join(", ", distinctNames));
            Console.ResetColor();

            Console.ReadLine();
        }

        //get text input
        private static string GetTextInput()
        {
            Console.ResetColor();
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("Enter text to extract human names (press 'Enter' twice to finish):");

            StringBuilder inputText = new StringBuilder();
            string line;
            while ((line = Console.ReadLine()) != null && line != "")
            {
                inputText.AppendLine(line);
            }

            return inputText.ToString();
        }
    }
}