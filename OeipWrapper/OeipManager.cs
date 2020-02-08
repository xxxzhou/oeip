using OeipCommon;
using OeipWrapper.Live;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    public class OeipManager : MSingleton<OeipManager>
    {
        //专门搞个字段,是为了避免这个传递给C++的回调会被GC给回收,故在这里传递给C++的回调都遵从这种设计
        private OnLogDelegate onLogDelegate;
        public event OnLogDelegate OnLogEvent;

        private OeipLiveContext liveCtx = new OeipLiveContext();
        /// <summary>
        /// 高版本可以使用ref 局部变量结构来方便更新类里的结构
        /// https://docs.microsoft.com/zh-cn/dotnet/csharp/language-reference/keywords/ref?f1url=https%3A%2F%2Fmsdn.microsoft.com%2Fquery%2Fdev16.query%3FappId%3DDev16IDEF1%26l%3DZH-CN%26k%3Dk(ref_CSharpKeyword)%3Bk(SolutionItemsProject)%3Bk(TargetFrameworkMoniker-.NETFramework%2CVersion%3Dv4.6.1)%3Bk(DevLang-csharp)%26rd%3Dtrue
        /// </summary>
        public ref OeipLiveContext LiveCtx
        {
            get
            {
                return ref liveCtx;
            }
        }
        //public List<OeipPipe> OeipPipes { get; private set; } = new List<OeipPipe>();
        public List<OeipDeviceInfo> OeipDevices { get; private set; } = new List<OeipDeviceInfo>();

        protected override void Init()
        {
            onLogDelegate = new OnLogDelegate(OnLogHandle);
            OeipHelper.setLogAction(onLogDelegate);
            OeipHelper.initOeip();
            this.GetCameras();

            LiveCtx.liveMode = OeipLiveMode.OIEP_FFMPEG;
            LiveCtx.bLoopback = 0;
            LiveCtx.liveServer = "http://129.211.40.225:6110";//"http://129.211.40.225:6110" "http://127.0.0.1:6110"
        }

        public bool IsCudaLoad
        {
            get
            {
                return OeipHelper.bCuda();
            }
        }

        /// <summary>
        /// 得到所有Camera
        /// </summary>
        private void GetCameras()
        {
            OeipDevices.Clear();
            int count = OeipHelper.getDeviceCount();
            if (count <= 0)
                return;
            int deviceLenght = Marshal.SizeOf(typeof(OeipDeviceInfo));
            var devices = PInvokeHelper.GetPInvokeArray<OeipDeviceInfo>(count,
                (IntPtr ptr, int pcount) =>
                {
                    OeipHelper.getDeviceList(ptr, pcount);
                });
            OeipDevices = devices.ToList();
        }
        /// <summary>
        /// 得到OeipCamera类型或是对应子类
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="cameraId"></param>
        /// <returns></returns>
        public T GetCamera<T>(int cameraId) where T : OeipCamera, new()
        {
            if (cameraId < 0 && cameraId >= OeipDevices.Count)
                return null;
            T camera = new T();
            camera.SetDevice(OeipDevices[cameraId]);
            return camera;
        }

        public List<VideoFormat> GetCameraFormats(int cameraId)
        {
            if (cameraId < 0 && cameraId >= OeipDevices.Count)
                return null;
            int count = OeipHelper.getFormatCount(cameraId);
            if (count > 0)
            {
                var videoFormats = PInvokeHelper.GetPInvokeArray<VideoFormat>(count,
                (IntPtr ptr, int pcount) =>
                {
                    OeipHelper.getFormatList(cameraId, ptr, pcount);
                });
                return videoFormats.ToList();
            }
            return new List<VideoFormat>();
        }

        public T CreatePipe<T>(OeipGpgpuType oeipGpgpuType) where T : OeipPipe, new()
        {
            int pipeId = OeipHelper.initPipe(oeipGpgpuType);
            if (pipeId < 0)
                return null;
            T pipe = new T();
            pipe.SetPipeId(pipeId);
            //OeipPipes.Add(pipe);
            return pipe;
        }

        private void OnLogHandle(int level, string message)
        {
            OnLogEvent?.Invoke(level, message);
        }

        public override void Close()
        {
            OeipHelper.shutdownOeip();
        }
    }
}
