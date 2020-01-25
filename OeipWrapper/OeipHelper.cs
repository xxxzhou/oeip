using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    #region OEIP enum/struct 
    public enum OeipVideoType
    {
        OEIP_VIDEO_OTHER,
        OEIP_VIDEO_NV12,//OEIP_YUVFMT_YUV420SP
        OEIP_VIDEO_YUY2,//OEIP_YUVFMT_YUY2I
        OEIP_VIDEO_YVYU,//OEIP_YUVFMT_YVYUI
        OEIP_VIDEO_UYVY,//OEIP_YUVFMT_UYVYI
        //设定自动编码成OEIP_VIDEO_YUY2
        OEIP_VIDEO_MJPG,
        OEIP_VIDEO_RGB24,
        OEIP_VIDEO_BGRA32,
        OEIP_VIDEO_RGBA32,
        OEIP_VIDEO_DEPTH,//U16
    }

    //Planar(YUV各自分开)Semi-Planar(Y单独分开,UV交并)Interleaved(YUV交并)
    public enum OeipYUVFMT
    {
        OEIP_YUVFMT_OTHER,
        OEIP_YUVFMT_YUV420SP,//Semi-Planar 一般用于图像设备 NV12
        OEIP_YUVFMT_YUY2I,//Interleaved 一般用于图像设备
        OEIP_YUVFMT_YVYUI,//Interleaved 一般用于图像设备
        OEIP_YUVFMT_UYVYI,//Interleaved 一般用于图像设备
        OEIP_YUVFMT_YUY2P,//Planar 一般用于传输
        OEIP_YUVFMT_YUV420P,//Planar 一般用于传输
    }

    public enum OeipLayerType
    {
        OEIP_NONE_LAYER,
        OEIP_INPUT_LAYER,
        OEIP_OUTPUT_LAYER,
        OEIP_YUV2RGBA_LAYER,
        OEIP_MAPCHANNEL_LAYER,
        OEIP_RGBA2YUV_LAYER,
        OEIP_RESIZE_LAYER,
        OEIP_OPERATE_LAYER,
        OEIP_BLEND_LAYER,
        OEIP_GUIDEDFILTER_LAYER,
        //此层计划只有CUDA的实现，DX11实现太麻烦了
        OEIP_GRABCUT_LAYER,
        OEIP_DARKNET_LAYER,
        OEIP_MAX_LAYER,
    };

    public enum OeipGpgpuType
    {
        OEIP_GPGPU_OTHER,
        OEIP_DX11,
        OEIP_CUDA,
        OEIP_Vulkun,
    }

    public enum OeipDeviceEventType
    {
        OEIP_Event_Other,
        //关闭成功
        OEIP_DeviceStop,
        //打开成功
        OEIP_DeviceOpen,
        //打开失败
        OEIP_DeviceNoOpen,
        //掉线
        OEIP_DeviceDropped,
    }

    public enum VideoDeviceType
    {
        OEIP_VideoDevice_Other,
        OEIP_MF,
        OEIP_Decklink,
        OEIP_Realsense,
        OEIP_Virtual,
    }

    public enum OeipAudioDataType
    {
        //静音
        OEIP_AudioData_None = 0,
        OEIP_Audio_Data,
        OEIP_Audio_WavHeader,
    }

    public struct OeipRect
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
    };

    public struct InputParamet
    {
        public int bCpu;// = true;
        public int bGpu;// = false;
    }

    public struct OutputParamet
    {
        public int bCpu;// = true;
        public int bGpu;// = true;
    }

    public struct YUV2RGBAParamet
    {
        public OeipYUVFMT yuvType;// = OEIP_YUVFMT_YUV420SP;
    }

    public struct RGBA2YUVParamet
    {
        public OeipYUVFMT yuvType;// = OEIP_YUVFMT_YUV420SP;
    }

    public struct MapChannelParamet
    {
        public int red;// = 0;
        public int green;// = 1;
        public int blue;// = 2;
        public int alpha;// = 3;
    }

    public struct ResizeParamet
    {
        public int bLinear;// = true;
        public int width;// = 1920;
        public int height;// = 1080;      
    }

    public struct OperateParamet
    {
        public int bFlipX;// = false;
        public int bFlipY;// = false;
        public float gamma;// = 1.0f;      
    }

    public struct BlendParamet
    {
        //所有值范围在0.1
        public OeipRect rect;
        //不透明度
        public float opacity;// = 0.f;
    }

    public struct GuidedFilterParamet
    {
        public int zoom;//= 8;
        public int softness;//= 5;
        public float eps;// = 0.00001f;
        public float intensity;// = 0.2f;
    }

    public struct GrabcutParamet
    {
        //画背景点或是grabcut扣像
        public int bDrawSeed;//= false;
        //是否使用GPU来计算一桢的高斯混合模型
        public int bGpuSeed;//= false;
        public int iterCount;// = 1;
        public int seedCount;// = 1000; 
        public int count;// = 250;
        public float gamma;// = 90.f;
        public float lambda;// = 450.f;
        public OeipRect rect;// = { };
    };

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct DarknetParamet
    {
        public int bLoad;//= false;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string confile;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string weightfile;
        public float thresh;// = 0.3f;
        public float nms;// = 0.4f;
        public int bDraw;//=false;
        public uint drawColor;//255
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PersonBox
    {
        //是人的概率
        public float prob;//= 0.f;
        public OeipRect rect;
    };

    public struct VideoFormat
    {
        public int index;// = -1;
        public int width;//= 0;
        public int height;// = 0;
        public OeipVideoType videoType;// = OEIP_VIDEO_OTHER;
        public int fps;// = 0;

        public string GetVideoType()
        {
            string strVideo = videoType.ToString().Replace("OEIP_VIDEO_", "");
            return strVideo;
        }
    };

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Parametr
    {
        public int CurrentValue;
        public int Min;
        public int Max;
        public int Step;
        public int Default;
        public int Flag;
    }

    [Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct CamParametrs
    {
        public Parametr Brightness;
        public Parametr Contrast;
        public Parametr Hue;
        public Parametr Saturation;
        public Parametr Sharpness;
        public Parametr Gamma;
        public Parametr ColorEnable;
        public Parametr WhiteBalance;
        public Parametr BacklightCompensation;
        public Parametr Gain;

        public Parametr Pan;
        public Parametr Tilt;
        public Parametr Roll;
        public Parametr Zoom;
        public Parametr Exposure;
        public Parametr Iris;
        public Parametr Focus;
    }

    /// <summary>
    /// 摄像机设备信息
    /// </summary>
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct OeipDeviceInfo
    {
        public int id;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string deviceName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string deviceId;

        public override string ToString()
        {
            if (id < 0)
                return "none";
            return id + " " + deviceName;
        }
    }

    public struct OeipAudioDesc
    {
        public int channel;//1,2
        public int sampleRate;//8000,11025,22050,44100
        public int bitSize;//16，24，32 
    }

    public struct OeipVideoDesc
    {
        public int width;
        public int height;
        public int fps;
        public OeipVideoType videoType;
    };

    public struct OeipDateDesc
    {
        public int elementSize;
        public int elementChannel;
    };
    #endregion

    #region callback
    /// <summary>
    /// 日志回调
    /// </summary>
    /// <param name="level"></param>
    /// <param name="message"></param>
    [UnmanagedFunctionPointer(PInvokeHelper.funcall, CharSet = CharSet.Ansi)]
    public delegate void OnLogDelegate(int level, string message);

    /// <summary>
    /// type指定事件类型，如设备，code指定类型结果，如设备中断等
    /// </summary>
    /// <param name="type"></param>
    /// <param name="code"></param>
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnEventDelegate(int type, int code);

    /// <summary>
    /// 摄像机的数据处理回调，dataType指明data数据类型
    /// </summary>
    /// <param name="data"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnReviceDelegate(IntPtr data, int width, int height);

    /// <summary>
    /// GPU运算管线返回
    /// </summary>
    /// <param name="layerIndex"></param>
    /// <param name="data"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="outputIndex"></param>
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnProcessDelegate(int layerIndex, IntPtr data, int width, int height, int outputIndex);
    #endregion

    public static class OeipHelper
    {
        public const string OeipDll = "oeip";

        #region oeip common
        /// <summary>
        /// 打印oeip产生的各种日志信息
        /// </summary>
        /// <param name="handler"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setLogAction(OnLogDelegate handler);

        /// <summary>
        /// 初始化OEIP环境
        /// </summary>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void initOeip();

        /// <summary>
        /// 销毁OEIP产生的各种资源
        /// </summary>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void shutdownOeip();

        /// <summary>
        /// 根据设备的视频类型返回对应YUV类型，如果非YUV类型，返回OEIP_YUVFMT_OTHER
        /// </summary>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern OeipYUVFMT getVideoYUV(OeipVideoType videoType);

        /// <summary>
        /// 相应颜色参数一般用uint来表示，用来给用户根据各个通道分量生成uint颜色，通道分量范围0.f-1.f
        /// </summary>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern uint getColor(float r, float g, float b, float a);

        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool bCuda();
        #endregion

        #region camera device
        /// <summary>
        /// 返回所有设备的个数
        /// </summary>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int getDeviceCount();

        /// <summary>
        /// 返回设备信息列表
        /// </summary>
        /// <param name="deviceList"></param>
        /// <param name="lenght"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int getDeviceList(IntPtr deviceList, int lenght, int index = 0);

        /// <summary>
        /// 捕获视频设备的图像格式数量
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int getFormatCount(int deviceIndex);

        /// <summary>
        /// 得到捕获视频设备当前所用的图像格式索引
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <param name="formatList"></param>
        /// <param name="lenght"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int getFormatList(int deviceIndex, IntPtr formatList, int lenght, int index = 0);

        /// <summary>
        /// 得到捕获视频设备当前所用的图像格式索引
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int getFormat(int deviceIndex);

        /// <summary>
        /// 捕获视频设备设置对应格式
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setFormat(int deviceIndex, int formatIndex);

        /// <summary>
        /// 打开设备
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool openDevice(int deviceIndex);

        /// <summary>
        /// 关闭设备
        /// </summary>
        /// <param name="deviceIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void closeDevice(int deviceIndex);

        /// <summary>
        /// 是否打开
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool bOpen(int deviceIndex);

        /// <summary>
        /// 查找设备最适合选定的长宽的格式索引
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fps"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int findFormatIndex(int deviceIndex, int width, int height, int fps = 30);

        /// <summary>
        /// 返回摄像机的内部参数设置
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <param name="parametrs"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void getDeviceParametrs(int deviceIndex, out CamParametrs parametrs);

        /// <summary>
        /// 更新摄像机的内部参数设置
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <param name="parametrs"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setDeviceParametrs(int deviceIndex, ref CamParametrs parametrs);


        /// <summary>
        /// 设置捕获视频设备每桢处理完后的数据回调，回调包含长，宽，数据指针
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <param name="onProcessData"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setDeviceDataAction(int deviceIndex, OnReviceDelegate onProcessData);

        /// <summary>
        /// 设置捕获视频设备事件回调，如没有正常打开,意外断掉等。
        /// </summary>
        /// <param name="deviceIndex"></param>
        /// <param name="onProcessData"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setDeviceEventAction(int deviceIndex, OnEventDelegate onDeviceEvent);

        #endregion

        #region gpgpu pipe
        /// <summary>
        /// 初始化一个GPU计算管线
        /// </summary>
        /// <param name="gpgpuType"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern int initPipe(OeipGpgpuType gpgpuType);

        /// <summary>
        /// 释放一个GPU计算管线,相应pipeId在没有再次initPipe得到,不能调用下面运用管线的API
        /// </summary>
        /// <param name="pipeId"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool closePipe(int pipeId);

        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool emptyPipe(int pipeId);

        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern OeipGpgpuType getPipeType(int pipeId);

        /// <summary>
        /// 管线添加一层,paramet表示管线对应的参数结构,请传递对应结构
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerName"></param>
        /// <param name="layerType"></param>
        /// <param name="paramet"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall, CharSet = CharSet.Ansi)]
        public static extern int addPiepLayer(int pipeId, string layerName, OeipLayerType layerType, IntPtr paramet);

        /// <summary>
        /// 连接二层数据处理层，注意上一层的输出格式要与下一层的输入格式对应，默认自动链接上一层，有分支时会使用
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="forwardName"></param>
        /// <param name="inputIndex"></param>
        /// <param name="selfIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall, CharSet = CharSet.Ansi)]
        public static extern void connectLayerName(int pipeId, int layerIndex, string forwardName, int inputIndex = 0, int selfIndex = 0);

        /// <summary>
        /// 同上，连接二层数据处理层，上一层数据用索引
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="forwardName"></param>
        /// <param name="inputIndex"></param>
        /// <param name="selfIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void connectLayerIndex(int pipeId, int layerIndex, int forwardIndex, int inputIndex = 0, int selfIndex = 0);

        /// <summary>
        /// 设定当前层是否可用
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="bEnable"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setEnableLayer(int pipeId, int layerIndex, bool bEnable);

        /// <summary>
        /// 设定当前层及关联这层的分支全部不可用
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="bEnable"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setEnableLayerList(int pipeId, int layerIndex, bool bEnable);

        /// <summary>
        /// 设置计算管线处理完后的数据回调，回调包含长，宽，数据指针，对应数据输出类型,用于C/C#使用
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="onProcessData"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setPipeDataAction(int pipeId, OnProcessDelegate onProcessData);

        /// <summary>
        /// 设置计算管线的输入
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="dataType"></param>
        /// <param name="inputIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void setPipeInput(int pipeId, int layerIndex, int width, int height, int dataType = 0, int inputIndex = 0);
        /// <summary>
        /// 更新计算管线的数据输入
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="data"></param>
        /// <param name="inputIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void updatePipeInput(int pipeId, int layerIndex, IntPtr data, int inputIndex = 0);
        /// <summary>
        /// 运行管线，如果为false,则查找日志输出信息
        /// </summary>
        /// <param name="pipeId"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool runPipe(int pipeId);
        /// <summary>
        /// 把另一个DX11上下文中的纹理当做当前管线的输入源
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="device"></param>
        /// <param name="tex"></param>
        /// <param name="inputIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void updatePipeInputGpuTex(int pipeId, int layerIndex, IntPtr device, IntPtr tex, int inputIndex = 0);
        /// <summary>
        /// 把当前管线的输出结果直接放入另一个DX11上下文的纹理中
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="device"></param>
        /// <param name="tex"></param>
        /// <param name="outputIndex"></param>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void updatePipeOutputGpuTex(int pipeId, int layerIndex, IntPtr device, IntPtr tex, int outputIndex = 0);
        /// <summary>
        /// 更新当前层的参数，需要注意paramet是当前层的参数结构，不同会引发想不到的问题
        /// </summary>
        /// <param name="pipeId"></param>
        /// <param name="layerIndex"></param>
        /// <param name="paramet"></param>
        /// <returns></returns>
        [DllImport(OeipDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern unsafe bool updatePipeParamet(int pipeId, int layerIndex, void* paramet);
        #endregion

    }
}
