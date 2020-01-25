using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    /// <summary>
    /// unity3D在托管代码中，拿不到DX11设备，此类封装Unity3D里的DX11设备。
    /// </summary>
    public class OeipHelperUnity3D
    {
        public const string OeipUnity3D = "oeip-unity3d";
        /// <summary>
        /// 传入输入的纹理
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="tex"></param>
        /// <param name="inputIndex"></param>
        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern void SetPipeInputGpuTex(int pipeId, int layerIndex, IntPtr tex, int inputIndex = 0);
        /// <summary>
        /// 传入输出的纹理
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="tex"></param>
        /// <param name="outputIndex"></param>
        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern void SetPipeOutputGpuTex(int pipeId, int layerIndex, IntPtr tex, int outputIndex = 0);
        /// <summary>
        /// 请这样调用GL.IssuePluginEvent(OeipHelperUnity3D.GetUpdateTexFunc(), pipeId),确保在渲染线程下更新当前
        /// 管线下的所有输出纹理
        /// </summary>
        /// <returns></returns>
        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern IntPtr GetUpdateTexFunc();
        /// <summary>
        /// 请这样调用GL.IssuePluginEvent(OeipHelperUnity3D.SetUpdateTexFunc(), pipeId),确保在渲染线程下更新当前
        /// 管线下的所有输入纹理
        /// </summary>
        /// <returns></returns>
        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern IntPtr SetUpdateTexFunc();
    }
}
