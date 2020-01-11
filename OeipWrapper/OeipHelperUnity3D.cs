using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    //如果是在unity3D下，传入传出DX11纹理请用这里
    //runPipe请改为GL.IssuePluginEvent(OeipHelperUnity3D.GetRunPipeFunc(), pipeId),确保在渲染线程下运行
    public class OeipHelperUnity3D
    {
        public const string OeipUnity3D = "oeip-unity3d";

        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern void SetPipeInputGpuTex(int pipeId, int layerIndex, IntPtr tex, int inputIndex = 0);

        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern void SetPipeOutputGpuTex(int pipeId, int layerIndex, IntPtr tex, int outputIndex = 0);

        [DllImport(OeipUnity3D, CallingConvention = PInvokeHelper.stdCall)]
        public static extern IntPtr GetRunPipeFunc();
    }
}
