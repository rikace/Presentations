using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CSMisc
{
    public class CleanNames
    {
        public string CleanNamesj(List<string> listOfNames)
        {
            StringBuilder resutl = new StringBuilder();
            for (int i = 0; i < listOfNames.Count; i++)
            {
                if (listOfNames[i].Length > 1)
                {
                    resutl.Append(CapitalizeString(listOfNames[i])).Append(",");
                }
            }
            return resutl.ToString().Substring(0, resutl.Length - 1);
        }

        public string CleanNamesLing(List<string> listOfNames)
        {
            var result = (from s in listOfNames
                where s.Length > 1
                select CapitalizeString(s))
                .Aggregate((new StringBuilder()), (acc, s) =>
                    acc.Append(s).Append(","));
            return result.ToString();
        }

        private string CapitalizeString(string s)
        {
            return s.Substring(0, 1).ToUpper() + s.Substring(1, s.Length);
        }

        public Dictionary<string, int> SumWords(string text)
        {
            string[] keywords = new[] {"in", "the", "then", "than", "a"};

            var frequencyList = text.Split('\n')
                .Select(c => c.ToLower())
                .Where(c => !keywords.Contains(c))
                .GroupBy(c => c)
                .Select(g => new {Word = g.Key, Count = g.Count()});

            return frequencyList.ToDictionary(k => k.Word, v => v.Count);
        }
    }
}
       