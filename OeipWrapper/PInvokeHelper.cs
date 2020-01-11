using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    public static class PInvokeHelper
    {
        public const CallingConvention stdCall = CallingConvention.StdCall;
        public const CallingConvention cdeclCall = CallingConvention.Cdecl;

#if OEIPSHARP_ANDROID
        public const CallingConvention funcall = CallingConvention.StdCall;
#else
        public const CallingConvention funcall = CallingConvention.Cdecl;
#endif

        public static T[] GetPInvokeArray<T>(int count, Action<IntPtr, int> cinvokeAction) where T : struct
        {
            //后期切换成net core,可以用Span类型来自动处理这些            
            T[] objs = new T[count];
            int objLenght = Marshal.SizeOf(typeof(T));
            byte[] data = new byte[objLenght * count];
            GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
            try
            {
                IntPtr pin = handle.AddrOfPinnedObject();
                cinvokeAction(pin, count);
                for (int i = 0; i < count; i++)
                {
                    objs[i] = ByteArrayToStructure<T>(pin, i * objLenght);
                }
            }
            finally
            {
                handle.Free();
            }
            return objs;
        }

        public static T ByteArrayToStructure<T>(IntPtr pin, int offset) where T : struct
        {
            try
            {
                return (T)Marshal.PtrToStructure(new IntPtr(pin.ToInt64() + offset), typeof(T));
            }
            catch (Exception)
            {
                return default(T);
            }
        }
    }
}
