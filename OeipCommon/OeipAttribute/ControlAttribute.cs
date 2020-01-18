using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace OeipCommon.OeipAttribute
{
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property)]
    public class ControlAttribute : Attribute
    {
        public string DisplayName { get; set; } = string.Empty;
        public int Order { get; set; } = -1;
        public MemberInfo Member { get; set; } = null;

        public void SetValue<T, U>(ref T obj, U member)
        {
            //结构装箱，不然SetValue自动装箱的值拿不到
            object o = obj;
            if (Member is FieldInfo)
            {
                var field = Member as FieldInfo;
                if (typeof(U) != field.FieldType)
                {
                    object ov = OeipAttributeHelper.ChangeType(member, field.FieldType);
                    field.SetValue(o, ov);
                }
                else
                {
                    field.SetValue(o, member);
                }
            }
            obj = (T)o;
        }

        public T GetValue<T>(ref object obj)
        {
            T t = default(T);
            if (Member is FieldInfo)
            {
                var field = Member as FieldInfo;
                var tv = field.GetValue(obj);
                if (typeof(T) == field.FieldType)
                    t = (T)tv;
                else
                    t = (T)OeipAttributeHelper.ChangeType(tv, typeof(T));
            }
            return t;
        }
    }

    public class SliderInputAttribute : ControlAttribute
    {
        public bool IsAutoRange { get; set; } = false;
        public bool IsInt { get; set; } = false;
        public float Range { get; set; } = 1;
        public float Min { get; set; } = 0;
        public float Max { get; set; } = 1;
        public float Default { get; set; } = 0;
    }

    public class InputAttribute : ControlAttribute
    {
        public string Default { get; set; } = string.Empty;
    }

    public class ToggleAttribute : ControlAttribute
    {
        public bool Default { get; set; } = false;
    }

    public class DropdownAttribute : ControlAttribute
    {
        public bool IsAutoDefault { get; set; } = false;
        public string Parent { get; set; } = string.Empty;
        public int Default { get; set; } = 0;
    }
}
