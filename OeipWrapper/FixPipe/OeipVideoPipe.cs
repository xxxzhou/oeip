using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.FixPipe
{
    /// <summary>
    /// 主要处理从设备读取数据展示，包含CUDA模式下的Darknet显示
    /// </summary>
    public class OeipVideoPipe : IDXViewPipe
    {
        public OeipPipe Pipe { get; private set; }
        public int InputIndex { get; private set; }
        public int Yuv2Rgba { get; private set; }
        public int Rgba2Yuv { get; private set; }
        public int MapChannel { get; private set; }
        public int OutMap { get; private set; }
        public int OutIndex { get; private set; }
        public int OutYuvIndex { get; private set; }
        public int ResizeIndex { get; private set; }
        public int DarknetIndex { get; private set; }
        public int GrabcutIndex { get; private set; }
        public int MattingOutIndex { get; private set; }

        public int OutGpuIndex
        {
            get
            {
                if (Pipe.GpgpuType == OeipGpgpuType.OEIP_CUDA)
                    return MattingOutIndex;
                return MattingOutIndex;
            }
        }
        /// <summary>
        /// 是否返回CPU数据
        /// </summary>
        public bool IsCpu { get; private set; } = false;
        /// <summary>
        /// 是否返回GPU数据
        /// </summary>
        public bool IsGpu { get; private set; } = true;

        public OeipYUVFMT YUVFMT { get; set; } = OeipYUVFMT.OEIP_YUVFMT_YUV420P;

        private DarknetParamet darknetParamet = new DarknetParamet();
        private GrabcutParamet grabcutParamet = new GrabcutParamet();

        public OeipVideoPipe(OeipPipe pipe)
        {
            this.Pipe = pipe;
            //添加输入层
            InputIndex = pipe.AddLayer("input", OeipLayerType.OEIP_INPUT_LAYER);
            //如果输入格式是YUV
            Yuv2Rgba = pipe.AddLayer("yuv2rgba", OeipLayerType.OEIP_YUV2RGBA_LAYER);
            //如果输入格式是BGR
            MapChannel = pipe.AddLayer("map channel", OeipLayerType.OEIP_MAPCHANNEL_LAYER);
            //mapChannel与yuv2rgba同级
            pipe.ConnectLayer(MapChannel, InputIndex);
            //经过大小变化
            ResizeIndex = pipe.AddLayer("resize", OeipLayerType.OEIP_RESIZE_LAYER);
            //如果显示格式要求BRG
            OutMap = pipe.AddLayer("out map channel", OeipLayerType.OEIP_MAPCHANNEL_LAYER);
            //输出层
            OutIndex = pipe.AddLayer("out put", OeipLayerType.OEIP_OUTPUT_LAYER);
            SetOutput(IsGpu, IsCpu);
            OutputParamet outputParamet = new OutputParamet();

            //输出YUV格式，为了推出流
            Rgba2Yuv = pipe.AddLayer("rgba2yuv", OeipLayerType.OEIP_RGBA2YUV_LAYER);
            RGBA2YUVParamet yuvParamet = new RGBA2YUVParamet();
            yuvParamet.yuvType = YUVFMT;
            Pipe.UpdateParamet(Rgba2Yuv, yuvParamet);
            //输出第二个流,YUV流,用于推送数据
            pipe.ConnectLayer(Rgba2Yuv, ResizeIndex);
            OutYuvIndex = pipe.AddLayer("out put yuv", OeipLayerType.OEIP_OUTPUT_LAYER);
            outputParamet.bGpu = 0;
            outputParamet.bCpu = 1;
            Pipe.UpdateParamet(OutYuvIndex, outputParamet);
            if (pipe.GpgpuType == OeipGpgpuType.OEIP_DX11)
            {
                int yuv2rgba2 = pipe.AddLayer("yuv2rgba 2", OeipLayerType.OEIP_YUV2RGBA_LAYER);
                YUV2RGBAParamet yparamet = new YUV2RGBAParamet();
                yparamet.yuvType = YUVFMT;
                Pipe.UpdateParamet(yuv2rgba2, yparamet);
                MattingOutIndex = pipe.AddLayer("matting out put", OeipLayerType.OEIP_OUTPUT_LAYER);
                outputParamet.bGpu = 1;
                outputParamet.bCpu = 0;
                Pipe.UpdateParamet(MattingOutIndex, outputParamet);
            }

            if (pipe.GpgpuType == OeipGpgpuType.OEIP_CUDA)
            {
                //神经网络层
                DarknetIndex = pipe.AddLayer("darknet", OeipLayerType.OEIP_DARKNET_LAYER);
                pipe.ConnectLayer(DarknetIndex, ResizeIndex);
                darknetParamet.bLoad = 1;
                darknetParamet.confile = "../../ThirdParty/yolov3-tiny-test.cfg";
                darknetParamet.weightfile = "../../ThirdParty/yolov3-tiny_745000.weights";
                darknetParamet.thresh = 0.3f;
                darknetParamet.nms = 0.3f;
                darknetParamet.bDraw = 1;
                darknetParamet.drawColor = OeipHelper.getColor(0.1f, 1.0f, 0.1f, 0.1f);
                Pipe.UpdateParametStruct(DarknetIndex, darknetParamet);
                //Grab cut扣像层
                GrabcutIndex = pipe.AddLayer("grab cut", OeipLayerType.OEIP_GRABCUT_LAYER);
                grabcutParamet.bDrawSeed = 0;
                grabcutParamet.iterCount = 1;
                grabcutParamet.seedCount = 1000;
                grabcutParamet.count = 250;
                grabcutParamet.gamma = 90.0f;
                grabcutParamet.lambda = 450.0f;
                grabcutParamet.rect = new OeipRect();
                Pipe.UpdateParamet(GrabcutIndex, grabcutParamet);
                //输出第三个流，网络处理层流
                MattingOutIndex = pipe.AddLayer("matting out put", OeipLayerType.OEIP_OUTPUT_LAYER);
                outputParamet.bGpu = 1;
                outputParamet.bCpu = 0;
                Pipe.UpdateParamet(MattingOutIndex, outputParamet);
            }
        }

        public void SetOutput(bool bCpu = false, bool bGpu = true)
        {
            //返回CPU还是GPU数据
            IsCpu = bCpu;
            IsGpu = bGpu;
            OutputParamet outputParamet = new OutputParamet();
            outputParamet.bGpu = bGpu ? 1 : 0;
            outputParamet.bCpu = bCpu ? 1 : 0;
            Pipe.UpdateParamet(OutIndex, outputParamet);
        }

        public void SetVideoFormat(OeipVideoType videoType, int width, int height)
        {
            //YUV类型数据
            var yuvType = OeipHelper.getVideoYUV(videoType);
            YUV2RGBAParamet yuvParamet = new YUV2RGBAParamet();
            yuvParamet.yuvType = yuvType;
            Pipe.UpdateParamet(Yuv2Rgba, yuvParamet);

            int inputWidth = width;
            int inputHeight = height;
            Pipe.SetEnableLayer(Yuv2Rgba, true);
            Pipe.SetEnableLayer(MapChannel, false);
            Pipe.SetEnableLayer(ResizeIndex, false);
            OeipDataType dataType = OeipDataType.OEIP_CU8C1;
            if (yuvType == OeipYUVFMT.OEIP_YUVFMT_OTHER)
            {
                Pipe.SetEnableLayer(Yuv2Rgba, false);
                if (videoType == OeipVideoType.OEIP_VIDEO_BGRA32)
                {
                    Pipe.SetEnableLayer(MapChannel, true);
                    MapChannelParamet mapChannelParamet = new MapChannelParamet();
                    mapChannelParamet.red = 2;
                    mapChannelParamet.green = 1;
                    mapChannelParamet.blue = 0;
                    mapChannelParamet.alpha = 3;
                    Pipe.UpdateParamet(MapChannel, mapChannelParamet);
                }
                else if (videoType == OeipVideoType.OEIP_VIDEO_RGB24)
                {
                    dataType = OeipDataType.OEIP_CU8C3;
                }
            }
            else if (yuvType == OeipYUVFMT.OEIP_YUVFMT_YUV420SP || yuvType == OeipYUVFMT.OEIP_YUVFMT_YUV420P || yuvType == OeipYUVFMT.OEIP_YUVFMT_YUY2P)
            {
                dataType = OeipDataType.OEIP_CU8C1;
                inputHeight = height * 3 / 2;
                if (yuvType == OeipYUVFMT.OEIP_YUVFMT_YUY2P)
                {
                    inputHeight = height * 2;
                }
            }
            else if (yuvType == OeipYUVFMT.OEIP_YUVFMT_UYVYI || yuvType == OeipYUVFMT.OEIP_YUVFMT_YUY2I || yuvType == OeipYUVFMT.OEIP_YUVFMT_YVYUI)
            {
                dataType = OeipDataType.OEIP_CU8C4;
                inputWidth = width / 2;
            }
            Pipe.SetInput(InputIndex, inputWidth, inputHeight, dataType);
        }

        public void RunVideoPipe(IntPtr data)
        {
            Pipe.UpdateInput(InputIndex, data);
            Pipe.RunPipe();
        }

        public void ChangeGrabcutMode(bool bDrawSeed, ref OeipRect rect)
        {
            grabcutParamet.bDrawSeed = bDrawSeed ? 1 : 0;
            grabcutParamet.rect = rect;
            Pipe.UpdateParamet(GrabcutIndex, grabcutParamet);
        }

        public void UpdateVideoParamet(OeipVideoParamet paramet)
        {
            grabcutParamet.iterCount = paramet.iterCount;
            grabcutParamet.seedCount = paramet.seedCount;
            grabcutParamet.count = paramet.count;
            grabcutParamet.gamma = paramet.gamma;
            grabcutParamet.lambda = paramet.lambda;
            grabcutParamet.bGpuSeed = paramet.bGpuSeed ? 1 : 0;
            Pipe.UpdateParamet(GrabcutIndex, grabcutParamet);
        }

        public void UpdateDarknetParamet(ref DarknetParamet paramet)
        {
            darknetParamet = paramet;
            Pipe.UpdateParametStruct(DarknetIndex, darknetParamet);
        }
    }
}
