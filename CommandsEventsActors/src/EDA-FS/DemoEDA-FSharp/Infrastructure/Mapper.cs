using System.Linq;
using System.Reflection;

namespace DemoEDAFSharp.Infrastructure
{
    public static class Mapper
    {
        public static TDest Map<TSource, TDest>(TSource source)
            where TSource : class
            where TDest : class, new()
        {
            var destination = new TDest();
            foreach (
                PropertyInfo destProp in
                    typeof (TDest).GetProperties(BindingFlags.Public | BindingFlags.Instance).Where(p => p.CanWrite))
            {
                PropertyInfo sourceProp =
                    typeof (TSource).GetProperties(BindingFlags.Public | BindingFlags.Instance)
                        .FirstOrDefault(
                            p =>
                                p.Name.ToUpper().Equals(destProp.Name.ToUpper()) &&
                                p.PropertyType == destProp.PropertyType);

                if (sourceProp != null)
                {
                    destProp.SetValue(destination, sourceProp.GetValue(source, null), null);
                }
            }
            return destination;
        }
    }
}