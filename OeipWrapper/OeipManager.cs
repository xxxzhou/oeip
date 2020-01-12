using OeipCommon;
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

        public List<OeipDeviceInfo> OeipDevices { get; private set; } = new List<OeipDeviceInfo>();

        protected override void Init()
        {
            onLogDelegate = new OnLogDelegate(OnLogHandle);
            OeipHelper.setLogAction(onLogDelegate);
            OeipHelper.initOeip();
            this.GetCameras();
        }

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

        //得到OeipCamera类型或是对应子类
        public T GetCamera<T>(int cameraId) where T : OeipCamera, new()
        {
            if (cameraId < 0 && cameraId >= OeipDevices.Count)
                return null;
            T camera = new T();
            camera.SetDevice(OeipDevices[cameraId]);
            return camera;
        }

        public T CreatePipe<T>(OeipGpgpuType oeipGpgpuType) where T : OeipPipe, new()
        {
            int pipeId = OeipHelper.initPipe(oeipGpgpuType);
            T pipe = new T();
            pipe.SetPipeId(pipeId);
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
