using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ParallelizingFuzzyMatch
{
    public static class Data
    {
        private const string PathFileText = @"..\..\..\Data\Shakespeare";

        public const int Iterations = 5;

        public static List<string> WordsToSearch = new List<string> { "ENGLISH", "RICHARD", "STEALLING", "MAGIC", "STARS", "MOON", "CASTLE" };

        public static char[] Delimiters = { ' ', ',', '.', ';', ':', '-', '_', '/', '\u000A' };
        public static List<KeyValuePair<string, string>> GetTextFileUrls()
        {
            var baseUrl = "http://teknology360.com/data/Shakespeare/";

            var files = new List<string> { "allswellthatendswell",
                "amsnd", "antandcleo",
                "asyoulikeit", "comedyoferrors",
                "cymbeline", "hamlet",
                "henryiv1", "henryiv2",
                "henryv", "henryvi1",
                "henryvi2", "henryvi3",
                "henryviii", "juliuscaesar",
                "kingjohn", "kinglear",
                "loveslobourslost", "maan",
                "macbeth", "measureformeasure",
                "merchantofvenice", "othello",
                "richardii", "richardiii",
                "romeoandjuliet", "tamingoftheshrew",
                "tempest", "themwofw",
                "thetgofv", "timon",
                "titus", "troilusandcressida",
                "twelfthnight", "winterstale"
            };
            return files.Select(f => new KeyValuePair<string, string>(f, string.Format("{0}{1}.txt", baseUrl, f))).ToList();
        }

        public static IEnumerable<string> GetFiles(string path = PathFileText)
        {
            if (!Directory.Exists(path))
                throw new ArgumentException("path");
            return Directory.EnumerateFiles(path, "*.txt");
        }

        public static string[] GetWords()
        {
            var files = GetFiles();
            var words = from file in files
                        from lines in File.ReadAllLines(file)
                        from word in lines.Split(Delimiters)
                        select word.ToUpper();
            return words.ToArray();
        }

        private static string[] words;
        public static string[] Words
        {
            get
            {
                if (words == null)
                    words = GetWords();
                return words;
            }
        }
    }
}
