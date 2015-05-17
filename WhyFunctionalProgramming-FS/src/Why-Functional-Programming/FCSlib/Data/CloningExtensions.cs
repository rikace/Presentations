using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.Linq.Expressions;

namespace FCSlib.Data {
  public static class CloningExtensions {
    public static T CloneWith<T>(this T source, Dictionary<string, object> newValues) {
      return (T) GetCreator(typeof(T))(source, newValues);
    }

    static Dictionary<Type, Dictionary<string, Func<object, object>>> accessors =
      new Dictionary<Type, Dictionary<string, Func<object, object>>>( );

    static Func<object, object> GetAccessor(Type type, string valueName) {
      Func<object, object> result;
      Dictionary<string, Func<object, object>> typeAccessors;

      // Let's see if we have an accessor already for this type/valueName
      if (accessors.TryGetValue(type, out typeAccessors)) {
        if (typeAccessors.TryGetValue(valueName, out result))
          return result;
      }

      // okay, create one and store it for later
      result = CreateAccessor(type, valueName);
      if (typeAccessors == null) {
        typeAccessors = new Dictionary<string, Func<object, object>>( );
        accessors[type] = typeAccessors;
      }
      typeAccessors[valueName] = result;

      return result;
    }

    static MethodInfo getAccessorMethod;
    static MethodInfo GetAccessorMethod {
      get {
        if (getAccessorMethod == null) {
          getAccessorMethod = typeof(CloningExtensions).GetMethod("GetAccessor", BindingFlags.Static | BindingFlags.NonPublic);
        }
        return getAccessorMethod;
      }
    }

    static Func<object, object> CreateAccessor(Type type, string valueName) {
      var finfo = type.GetField(valueName);
      var param = Expression.Parameter(typeof(object), "o");
      Expression<Func<object, object>> exp =
        Expression.Lambda<Func<object, object>>(
        Expression.Convert(
        Expression.Field(Expression.Convert(param, type), finfo), typeof(object)),
        param);
      return exp.Compile( );
    }

    static Dictionary<Type, Func<object, Dictionary<string, object>, object>> creators =
      new Dictionary<Type, Func<object, Dictionary<string, object>, object>>( );

    static Func<object, Dictionary<string, object>, object> GetCreator(Type type) {
      Func<object, Dictionary<string, object>, object> result;
      if (creators.TryGetValue(type, out result))
        return result;

      result = CreateCreator(type);
      creators[type] = result;

      return result;
    }

    static V GetValueOrNull<K, V>(Dictionary<K, V> dict, K key) {
      // This is the call that Dictionary<K,V> doesn't have - 
      // return a value if the key exists, otherwise null. Of course
      // you can model the same thing with an if, but that's less performant
      // than TryGetValue, and it can't be used inline either.
      V result;
      if (dict.TryGetValue(key, out result))
        return result;
      return default(V);
    }

    static MethodInfo getValueOrNullStringObjectMethod;
    static MethodInfo GetValueOrNullStringObjectMethod {
      get {
        if (getValueOrNullStringObjectMethod == null) {
          var genericMethod =  typeof(CloningExtensions).GetMethod("GetValueOrNull", BindingFlags.Static | BindingFlags.NonPublic);
          getValueOrNullStringObjectMethod = genericMethod.MakeGenericMethod(typeof(string), typeof(object));
        }
        return getValueOrNullStringObjectMethod;
      }
    }

    static Func<object, Dictionary<string, object>, object> CreateCreator(Type type) {
      var ctors = type.GetConstructors( );
      if (ctors.Length > 1)
        throw new InvalidOperationException(String.Format("Can't clone type {0} because it has more than one constructor.", type));
      var ctor = ctors[0];
      var cparams = ctor.GetParameters( );
      var paramCount = cparams.Length;
      var paramArray = new Expression[paramCount];

      var sourceParam = Expression.Parameter(typeof(object), "s");
      var dictParam = Expression.Parameter(typeof(Dictionary<string, object>), "d");
      
      //GetValueOrNull(dictParam, paramName) ?? GetAccessor(type, paramName)(sourceParam)

      for (int i = 0; i < paramCount; i++) {
        ConstantExpression paramName = Expression.Constant(cparams[i].Name);
        paramArray[i] =
          Expression.Convert(
            Expression.Coalesce(
              Expression.Call(GetValueOrNullStringObjectMethod, dictParam, paramName),
              Expression.Invoke(
                Expression.Call(GetAccessorMethod, Expression.Constant(type), paramName),
                sourceParam)),
            cparams[i].ParameterType);
      }

      Expression<Func<object, Dictionary<string, object>, object>> exp =
        Expression.Lambda<Func<object, Dictionary<string, object>, object>>(
          Expression.New(ctor, paramArray),
          sourceParam, dictParam);

      return exp.Compile( );
    }


  }
}
