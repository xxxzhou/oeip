using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    public class OeipCamera
    {
        public int Id { get; private set; } = -1;
        public string DeviceName { get; private set; } = string.Empty;
        public string DeviceId { get; private set; } = string.Empty;

        public bool IsInit
        {
            get
            {
                return this.Id >= 0;
            }
        }

        public List<VideoFormat> VideoFormats { get; private set; } = new List<VideoFormat>();

        //专门搞个字段,是为了避免这个传递给C++的回调会被GC给回收
        protected OnEventDelegate onEventDelegate;
        public event OnEventDelegate OnDeviceEvent;

        protected OnReviceDelegate onReviceDelegate;
        public event OnReviceDelegate OnReviceEvent;

        public OeipCamera()
        {
            onEventDelegate = new OnEventDelegate(OnEventHandle);
            onReviceDelegate = new OnReviceDelegate(OnReviceHandle);
        }

        public void SetDevice(OeipDeviceInfo oeipDevice)
        {
            this.Id = oeipDevice.id;
            this.DeviceId = oeipDevice.deviceId;
            this.DeviceName = oeipDevice.deviceName;
            this.GetCameraFormatList(this.Id);
            OeipHelper.setDeviceDataAction(this.Id, onReviceDelegate);
            OeipHelper.setDeviceEventAction(this.Id, onEventDelegate);
        }

        public int GetFormat()
        {
            if (!IsInit)
                return -1;
            return OeipHelper.getFormat(this.Id);
        }

        public void SetFormat(int index)
        {
            if (!IsInit)
                return;
            OeipHelper.setFormat(Id, index);
        }

        public bool Open()
        {
            if (!IsInit)
                return false;
            return OeipHelper.openDevice(Id);
        }

        public void Close()
        {
            if (!IsInit)
                return;
            OeipHelper.closeDevice(Id);
        }

        public bool IsOpen
        {
            get
            {
                if (IsInit)
                    return OeipHelper.bOpen(Id);
                return false;
            }
        }

        public int FindFormatIndex(int width, int height, int fps = 30)
        {
            if (!IsInit)
                return -1;
            return OeipHelper.findFormatIndex(Id, width, height, fps);
        }

        public void GetCameraFormatList(int index)
        {
            if (!IsInit)
                return;
            int count = OeipHelper.getFormatCount(index);
            if (count > 0)
            {
                var videoFormats = PInvokeHelper.GetPInvokeArray<VideoFormat>(count,
                (IntPtr ptr, int pcount) =>
                {
                    OeipHelper.getFormatList(index, ptr, pcount);
                });
                VideoFormats = videoFormats.ToList();
            }
        }

        public override string ToString()
        {
            if (Id == -1)
                return "none";
            return Id + " " + DeviceName;
        }

        private void OnEventHandle(int type, int code)
        {
            OnDeviceEvent?.Invoke(type, code);
        }

        private void OnReviceHandle(IntPtr data, int width, int height)
        {
            OnReviceEvent?.Invoke(data, width, height);
        }
    }
}
