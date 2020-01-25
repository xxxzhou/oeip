using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.Live
{
    #region OEIPLIVE enum/struct 
    public enum OeipLiveMode
    {
        OIEP_FFMPEG,
    }

    public enum OeipLiveOperate
    {
        OEIP_LIVE_OPERATE_NONE,
        //初始化
        OEIP_LIVE_OPERATE_INIT,
        //是否已经得到媒体服务器地址
        OEIP_LIVE_OPERATE_MEDIASERVE,
        //推流与拉流打开
        OEIP_LIVE_OPERATE_OPEN,
        OEIP_LIVE_OPERATE_CLOSE,
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct OeipLiveContext
    {
        //采用那种直播SDK
        public OeipLiveMode liveMode;//OIEP_FFMPEG
        //是否采集声卡
        public int bLoopback;//0
        //如果有需要，填写直播服务器
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
        public string liveServer;
    }

    public struct OeipVideoEncoder
    {
        public int width;//1920
        public int height;//1080
        public int fps;//30
        public int bitrate;//4000000
        public OeipYUVFMT yuvType;//OEIP_YUVFMT_YUY2P
    };

    public struct OeipAudioEncoder
    {
        public int frequency;//8000
        public int channel;//1
        public int bitrate;//48000        
    };

    [StructLayout(LayoutKind.Sequential)]
    public struct OeipVideoFrame
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public IntPtr[] data;
        public uint dataSize;
        public ulong timestamp;
        public uint width;
        public uint height;
        public OeipYUVFMT fmt;//OEIP_YUVFMT_YUY2P
    };

    public struct OeipAudioFrame
    {
        public IntPtr data;
        public uint dataSize;
        public ulong timestamp;
        public uint sampleRate;// = 8000;
        public uint channels;// = 1;
        public uint bitDepth;// = 16;
    };

    public struct OeipPushSetting
    {
        public int bAudio;//true   
        public int bVideo;//true   
        public OeipVideoEncoder videoEncoder;//
        public OeipAudioEncoder audioEncoder;//
    }
    #endregion
    /// <summary>
    /// 是否成功初始化，coda少于0是异常情况，0是正常返回，>0正常情况附加信息
    /// </summary>
    /// <param name="code"></param>
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnInitRoomDelegate(int code);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnLoginRoomDelegate(int code, int userid);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnUserChangeDelegate(int userId, bool bAdd);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnStreamUpdateDelegate(int userId, int index, bool bAdd);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnVideoFrameDelegate(int userId, int index, OeipVideoFrame videoFrame);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnAudioFrameDelegate(int userId, int index, OeipAudioFrame audioFrame);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnLogoutRoomDelegate(int code);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall, CharSet = CharSet.Ansi)]
    public delegate void OnOperateResultDelegate(int operate, int code, string message);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnPushStreamDelegate(int index, int code);
    [UnmanagedFunctionPointer(PInvokeHelper.funcall)]
    public delegate void OnPullStreamDelegate(int userId, int index, int code);

    public struct LiveBackWrapper
    {
        public OnInitRoomDelegate onInitRoomDelegate;
        public OnLoginRoomDelegate onLoginRoomDelegate;
        public OnUserChangeDelegate onUserChangeDelegate;
        public OnStreamUpdateDelegate onStreamUpdateDelegate;
        public OnVideoFrameDelegate onVideoFrameDelegate;
        public OnAudioFrameDelegate onAudioFrameDelegate;
        public OnLogoutRoomDelegate onLogoutRoomDelegate;
        public OnOperateResultDelegate onOperateResultDelegate;
        public OnPushStreamDelegate onPushStreamDelegate;
        public OnPullStreamDelegate onPullStreamDelegate;
    }

    public static class OeipLiveHelper
    {
        public const string OeipLiveDll = "oeip-live";
        /// <summary>
        /// 初始化OEIP直播SDK
        /// </summary>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void initOeipLive();
        /// <summary>
        /// 销毁OEIP直播相应资源
        /// </summary>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void shutdownOeipLive();
        /// <summary>
        /// 初始化OEIP直播房间相关参数与回调
        /// </summary>
        /// <param name="liveCtx"></param>
        /// <param name="liveBack"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool initLiveRoomWrapper(ref OeipLiveContext liveCtx, ref LiveBackWrapper liveBack);
        /// <summary>
        /// 登陆房间
        /// </summary>
        /// <param name="roomName"></param>
        /// <param name="userId"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall, CharSet = CharSet.Ansi)]
        public static extern bool loginRoom(string roomName, int userId);
        /// <summary>
        /// 推流
        /// </summary>
        /// <param name="index"></param>
        /// <param name="setting"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool pushStream(int index, ref OeipPushSetting setting);
        /// <summary>
        /// 推视频数据
        /// </summary>
        /// <param name="index">第几部流</param>
        /// <param name="videoFrame"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool pushVideoFrame(int index, ref OeipVideoFrame videoFrame);
        /// <summary>
        /// 推音频数据
        /// </summary>
        /// <param name="index"></param>
        /// <param name="audioFrame"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool pushAudioFrame(int index, ref OeipAudioFrame audioFrame);
        /// <summary>
        /// 停止推流
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool stopPushStream(int index);
        /// <summary>
        /// 拉流
        /// </summary>
        /// <param name="userId"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool pullStream(int userId, int index);
        /// <summary>
        /// 停止拉流
        /// </summary>
        /// <param name="userId"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool stopPullStream(int userId, int index);
        /// <summary>
        /// 拿出房间
        /// </summary>
        /// <returns></returns>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern bool logoutRoom();
        /// <summary>
        /// 根据data,width,height,fmt组合一个OeipVideoFrame
        /// </summary>
        /// <param name="data"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <param name="fmt"></param>
        /// <param name="videoFrame"></param>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void getVideoFrame(IntPtr data, int width, int height, OeipYUVFMT fmt, ref OeipVideoFrame videoFrame);
        /// <summary>
        /// 复制OeipVideoFrame里数据到IntPtr
        /// </summary>
        /// <param name="data"></param>
        /// <param name="videoFrame"></param>
        [DllImport(OeipLiveDll, CallingConvention = PInvokeHelper.funcall)]
        public static extern void getVideoFrameData(IntPtr data, ref OeipVideoFrame videoFrame);

    }
}
