using OeipWrapper.Live;
using System;
using System.Runtime.InteropServices;

namespace OeipWrapper.FixPipe
{
    /// <summary>
    /// 用于拉流数据处理,主要处理YUV420P/YUV422P格式,请手动调用Dispose
    /// </summary>
    public class OeipLivePipe : IDXViewPipe, IDisposable
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
        private VideoFormat videoFormat = new VideoFormat();
        public ref VideoFormat VideoFormat
        {
            get
            {
                return ref videoFormat;
            }
        }
        private OeipYUVFMT yuvfmt = OeipYUVFMT.OEIP_YUVFMT_YUV420P;
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
            int inputHeight = VideoFormat.height * 2;
            OeipDataType dataType = OeipDataType.OEIP_CU8C1;
            if (yuvfmt == OeipYUVFMT.OEIP_YUVFMT_YUV420P)
            {
                inputHeight = VideoFormat.height * 3 / 2;
            }
            YUV2RGBAParamet paramet = new YUV2RGBAParamet();
            paramet.yuvType = yuvfmt;
            Pipe.UpdateParamet(Yuv2Rgba, paramet);
            Pipe.SetInput(InputIndex, VideoFormat.width, inputHeight, dataType);
            //重新申请复制当前视频桢数据空间
            FreeData();
            int size = VideoFormat.width * inputHeight;
            yuvData = Marshal.AllocHGlobal(size);
            OnLiveImageChange?.Invoke(videoFormat);
        }

        public void RunLivePipe(ref OeipVideoFrame videoFrame)
        {
            if (VideoFormat.width != videoFrame.width || VideoFormat.height != videoFrame.height || yuvfmt != videoFrame.fmt)
            {
                VideoFormat.fps = 30;
                VideoFormat.width = (int)videoFrame.width;
                VideoFormat.height = (int)videoFrame.height;
                VideoFormat.videoType = OeipVideoType.OEIP_VIDEO_RGBA32;
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
