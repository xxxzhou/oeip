using OeipCommon.OeipAttribute;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace OeipCommon
{
    public static class OeipAttributeHelper
    {
        public static List<ControlAttribute> GetAttributes(Type objType)
        {
            List<ControlAttribute> attributes = new List<ControlAttribute>();
            var members = objType.GetFields();
            foreach (var member in members)
            {
                if (!member.IsDefined(typeof(ControlAttribute), true))
                    continue;
                var autoAttributeList = member.GetCustomAttributes(typeof(ControlAttribute), true);
                if (autoAttributeList.Length > 0)
                {
                    ControlAttribute ba = (ControlAttribute)autoAttributeList[0];
                    ba.Member = member;
                    attributes.Add(ba);
                }
            }
            attributes.Sort((ControlAttribute a, ControlAttribute b) =>
            {
                if (a.Order > b.Order)
                    return 1;
                else if (a.Order < b.Order)
                    return -1;
                return 0;
            });
            return attributes;
        }

        public static List<ControlAttribute> GetAttributes<T>(T obj)
        {
            return GetAttributes(obj.GetType());
        }

        /// <summary>
        /// 强化版本的Convert.ChangeType，加强可空类型与Enum的处理
        /// </summary>
        /// <param name="obj"></param>
        /// <param name="conversionType"></param>
        /// <returns></returns>
        public static object ChangeType(object obj, Type conversionType)
        {
            return ChangeType(obj, conversionType, Thread.CurrentThread.CurrentCulture);
        }

        public static object ChangeType(object obj, Type conversionType, IFormatProvider provider)
        {
            #region Nullable
            Type nullableType = Nullable.GetUnderlyingType(conversionType);
            if (nullableType != null)
            {
                if (obj == null)
                {
                    return null;
                }
                return Convert.ChangeType(obj, nullableType, provider);
            }
            #endregion
            if (typeof(System.Enum).IsAssignableFrom(conversionType))
            {
                return Enum.Parse(conversionType, obj.ToString());
            }
            return Convert.ChangeType(obj, conversionType, provider);
        }
    }
}
