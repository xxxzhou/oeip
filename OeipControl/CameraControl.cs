using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OeipWrapper;
using OeipWrapper.FixPipe;

namespace OeipControl
{
    public partial class CameraControl : UserControl
    {
        private OeipCamera camera = null;
        private OeipVideoPipe videoPipe = null;
        private PersonBox[] personBox = null;
        public CameraControl()
        {
            InitializeComponent();
        }

        public void NativeLoad(OeipGpgpuType gpuType, int index = 0, bool bCpu = false)
        {
            var pipe = OeipManager.Instance.CreatePipe<OeipPipe>(gpuType);
            videoPipe = new OeipVideoPipe(pipe);
            videoPipe.Pipe.OnProcessEvent += Pipe_OnProcessEvent;
            videoPipe.SetOutput(bCpu, !bCpu);

            camera = OeipManager.Instance.GetCamera<OeipCamera>(index);
            camera.OnReviceEvent += Camera_OnReviceEvent;
            var cameraList = OeipManager.Instance.OeipDevices;
            OeipDeviceInfo nullDevice = new OeipDeviceInfo();
            nullDevice.id = -1;
            cbx_cameraList.Items.Add(nullDevice);
            foreach (var camera in cameraList)
            {
                cbx_cameraList.Items.Add(camera);
            }
            cbx_cameraList.SelectedIndex = Math.Min(index, cameraList.Count - 1);
        }

        private void Pipe_OnProcessEvent(int layerIndex, IntPtr data, int width, int height, int outputIndex)
        {
            if (layerIndex == videoPipe.OutIndex)
            {
                displayWF.UpdateImage(width, height, data);
            }
            else if (layerIndex == videoPipe.DarknetIndex)
            {
                personBox = PInvokeHelper.GetPInvokeArray<PersonBox>(width, data);
                if (width > 0)
                {
                    Action action = () =>
                    {
                        if (personBox == null)
                            return;
                        string msg = string.Empty;
                        foreach (var px in personBox)
                        {
                            msg += " " + px.prob;
                        }
                        this.label3.Text = $"人数:{width} {msg}";
                    };
                    this.BeginInvoke(action);
                }
            }
        }

        private void Camera_OnReviceEvent(IntPtr data, int width, int height)
        {
            videoPipe.RunVideoPipe(data);
        }

        private void cbx_cameraList_SelectedIndexChanged(object sender, EventArgs e)
        {
            cbx_formatList.Items.Clear();
            if (camera.IsOpen)
                camera.Close();
            var newCamera = (OeipDeviceInfo)cbx_cameraList.SelectedItem;
            if (newCamera.id < 0 || newCamera.id == camera.Id)
                return;
            camera.SetDevice(newCamera);
            foreach (var format in camera.VideoFormats)
            {
                cbx_formatList.Items.Add(format.width + "x" + format.height + " " + format.fps + "fps " + format.GetVideoType());
            }
            int formatIndex = camera.FindFormatIndex(1920, 1080);
            SetFormat(formatIndex);
        }

        private void cbx_formatList_SelectedIndexChanged(object sender, EventArgs e)
        {
            SetFormat(this.cbx_formatList.SelectedIndex);
        }

        public void SetFormat(int index)
        {
            var selectFormat = camera.VideoFormats[index];
            videoPipe.SetVideoFormat(selectFormat.videoType, selectFormat.width, selectFormat.height);

            cbx_formatList.SelectedIndex = index;
            camera.SetFormat(index);
            camera.Open();

            displayDx11.Visible = videoPipe.IsGpu;
            displayWF.Visible = !videoPipe.IsGpu;
            displayWF.Dock = DockStyle.Fill;
            displayDx11.Dock = DockStyle.Fill;
            displayDx11.NativeLoad(videoPipe, selectFormat);
        }
    }
}
