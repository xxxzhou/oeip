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
        /// <summary>
        /// C#申请空间，并交给非托管代码填充
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="count">T的个数</param>
        /// <param name="cinvokeAction">非托管代码如何填充IntPtr</param>
        /// <returns></returns>
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

        /// <summary>
        /// 把非托管数据转化成托管代码,C#本身不申请空间
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="count"></param>
        /// <param name="data"></param>
        /// <returns></returns>
        public static T[] GetPInvokeArray<T>(int count, IntPtr data) where T : struct
        {
            if (count <= 0)
                return null;
            int objLenght = Marshal.SizeOf(typeof(T));
            T[] objs = new T[count];
            for (int i = 0; i < count; i++)
            {
                objs[i] = ByteArrayToStructure<T>(data, objLenght * i);
            }
            return objs;
        }


        public static T ByteArrayToStructure<T>(IntPtr pin, int offset) where T : struct
        {
            try
            {                
                return (T)Marshal.PtrToStructure(pin + offset, typeof(T));
            }
            catch (Exception)
            {
                return default(T);
            }
        }
    }
}
