using OeipCommon;
using OeipWrapper;
using OeipWrapper.Live;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OeipWrapperTest
{
    public partial class LiveForm : Form
    {
        private bool bPush = false;
        //是否拉自己推的流,一般用来测试
        public bool IsPullSelf { get; set; } = true;
        public LiveForm()
        {
            InitializeComponent();
        }

        private void LiveForm_Load(object sender, EventArgs e)
        {
            OeipManager.Instance.OnLogEvent += Instance_OnLogEvent;
            cameraControl1.NativeLoad(OeipGpgpuType.OEIP_CUDA, 0, false);
            cameraControl1.VideoPipe.Pipe.OnProcessEvent += Pipe_OnProcessEvent;
            OeipLiveManager.Instance.OnLoginRoomEvent += Instance_OnLoginRoomEvent;
            OeipLiveManager.Instance.OnStreamUpdateEvent += Instance_OnStreamUpdateEvent;
            OeipLiveManager.Instance.OnVideoFrameEvent += Instance_OnVideoFrameEvent;
            liveControl1.NativeLoad(OeipGpgpuType.OEIP_CUDA);

            //新增加以纹理为输入的混合显示            
            blendControl1.NativeLoad(OeipGpgpuType.OEIP_CUDA, cameraControl1.Format);
            cameraControl1.SetBlendTex(blendControl1.BlendPipe, true);
            liveControl1.SetBlendTex(blendControl1.BlendPipe, false);
        }

        private void Instance_OnVideoFrameEvent(int userId, int index, OeipVideoFrame videoFrame)
        {
            this.liveControl1.LivePipe.RunLivePipe(ref videoFrame);
        }

        private void Instance_OnStreamUpdateEvent(int userId, int index, bool bAdd)
        {
            if (!IsPullSelf && OeipLiveManager.Instance.UserId == userId)
                return;
            if (bAdd)
            {
                OeipLiveManager.Instance.PullStream(userId, index);
            }
            else
            {
                OeipLiveManager.Instance.StopPullStream(userId, index);
            }
        }

        private void Pipe_OnProcessEvent(int layerIndex, IntPtr data, int width, int height, int outputIndex)
        {
            if (!bPush)
                return;
            if (layerIndex == cameraControl1.VideoPipe.OutYuvIndex)
            {
                OeipLiveManager.Instance.PushVideoFrame(0, data, width, height, this.cameraControl1.VideoPipe.YUVFMT);
            }
        }

        private void Instance_OnLogEvent(int level, string message)
        {
            OeipLogLevel oeipLogLevel = (OeipLogLevel)level;
            Action action = () =>
            {
                this.label1.Text = $"level:{oeipLogLevel} message:{message}";
            };
            this.BeginInvoke(action);
            LogHelper.LogMessage(message, oeipLogLevel);
        }

        private void Instance_OnLoginRoomEvent(int code, int userid)
        {
            OeipPushSetting pushSetting = new OeipPushSetting();
            pushSetting.bVideo = 1;
            pushSetting.bAudio = 0;
            bPush = OeipLiveManager.Instance.PushStream(0, ref pushSetting);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            int.TryParse(this.textBox2.Text, out int userId);
            OeipLiveManager.Instance.LoginRoom(this.textBox1.Text, userId);
        }

        private void LiveForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            cameraControl1.Close();
            OeipLiveManager.Instance.Close();
            OeipManager.Instance.Close();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            OeipLiveManager.Instance.LogoutRoom();
        }
    }
}
