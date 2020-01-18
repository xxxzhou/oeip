using OeipWrapper.Live;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.FixPipe
{
    public class OeipLivePipe : ISharpDXViewPipe, IDisposable
    {
        public event Action<VideoFormat> OnLiveImageChange;
        public OeipPipe Pipe { get; private set; }
        public int InputIndex { get; private set; }
        public int Yuv2Rgba { get; private set; }
        public int OutIndex { get; private set; }

        public bool IsGpu { get; private set; } = true;
        public int OutGpuIndex
        {
            get
            {
                return OutIndex;
            }
        }

        private uint width = 0;
        private uint height = 0;
        private OeipYUVFMT fmt = OeipYUVFMT.OEIP_YUVFMT_YUV420P;
        private IntPtr yuvData = IntPtr.Zero;
        public OeipLivePipe(OeipPipe pipe)
        {
            this.Pipe = pipe;

            //添加输入层
            InputIndex = pipe.AddLayer("input", OeipLayerType.OEIP_INPUT_LAYER);
            Yuv2Rgba = pipe.AddLayer("yuv2rgba", OeipLayerType.OEIP_YUV2RGBA_LAYER);
            OutIndex = pipe.AddLayer("out put", OeipLayerType.OEIP_OUTPUT_LAYER);

            OutputParamet outputParamet = new OutputParamet();
            outputParamet.bGpu = IsGpu ? 1 : 0;
            outputParamet.bCpu = 1;
            pipe.UpdateParamet(OutIndex, outputParamet);
        }

        private void FreeData()
        {
            if (yuvData != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(yuvData);
            }
        }

        private void ResetPipe()
        {
            int inputWidth = (int)width;
            int inputHeight = (int)height * 2;
            OeipDataType dataType = OeipDataType.OEIP_CU8C1;
            if (fmt == OeipYUVFMT.OEIP_YUVFMT_YUV420P)
            {
                inputHeight = (int)height * 3 / 2;
            }
            int size = inputWidth * inputHeight;
            Pipe.SetInput(InputIndex, inputWidth, inputHeight, dataType);
            //重新申请复制当前视频桢数据空间
            FreeData();
            yuvData = Marshal.AllocHGlobal(size);
            if (OnLiveImageChange != null)
            {
                VideoFormat videoFormat = new VideoFormat();
                videoFormat.fps = 30;
                videoFormat.width = (int)width;
                videoFormat.height = (int)height;
                OnLiveImageChange(videoFormat);
            }
        }

        public void RunLivePipe(ref OeipVideoFrame videoFrame)
        {
            if (width != videoFrame.width || height != videoFrame.heigh || fmt != videoFrame.fmt)
            {
                width = videoFrame.width;
                height = videoFrame.heigh;
                fmt = videoFrame.fmt;
                ResetPipe();
            }
            OeipLiveHelper.getVideoFrameData(yuvData, ref videoFrame);
            Pipe.UpdateInput(InputIndex, yuvData);
            Pipe.RunPipe();
        }

        public void Dispose()
        {
            if (yuvData != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(yuvData);
            }
        }
    }
}
