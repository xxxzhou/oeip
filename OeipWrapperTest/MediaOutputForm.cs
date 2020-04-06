using OeipCommon;
using OeipWrapper;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OeipWrapperTest
{
    public partial class MediaOutputForm : Form
    {
        private OeipMediaOutput mediaOutput = null;
        private OeipVideoFrame videoFrame = new OeipVideoFrame();
        private OeipAudioFrame audioFrame = new OeipAudioFrame();
        private OnAudioDataDelegate onAudioDataDelegate;
        //麦与声卡输出格式固定
        private OeipAudioDesc audioDesc = new OeipAudioDesc();

        private long openTime = 0;
        public MediaOutputForm()
        {
            InitializeComponent();
        }

        private void MediaOutputForm_Load(object sender, EventArgs e)
        {
            cameraControl1.NativeLoad(OeipGpgpuType.OEIP_CUDA, 0, false);
            cameraControl1.VideoPipe.Pipe.OnProcessEvent += Pipe_OnProcessEvent;
            onAudioDataDelegate = new OnAudioDataDelegate(onAudioDataAction);
            audioDesc.bitSize = 16;
            audioDesc.channel = 1;
            audioDesc.sampleRate = 22050;
            mediaOutput = OeipManager.Instance.GetMediaOutput();
        }

        private void Pipe_OnProcessEvent(int layerIndex, IntPtr data, int width, int height, int outputIndex)
        {
            if (!mediaOutput.IsOpen)
                return;
            if (layerIndex == cameraControl1.VideoPipe.OutYuvIndex)
            {
                OeipHelper.setVideoFrame(data, width, height, cameraControl1.VideoPipe.YUVFMT, ref videoFrame);
                videoFrame.timestamp = DateTime.Now.Ticks / 10000 - openTime;
                LogHelper.LogMessage("video time:" + videoFrame.timestamp);
                mediaOutput.PushVideoFrame(ref videoFrame);
            }
        }

        private void btn_openSrcFile_Click(object sender, EventArgs e)
        {
            var result = saveFileDialog1.ShowDialog();
            if (result != DialogResult.OK && result != DialogResult.Yes)
                return;
            var fileName = saveFileDialog1.FileName;
            this.textBox1.Text = fileName;
        }

        private void onAudioDataAction(IntPtr data, int legnth)
        {
            if (!mediaOutput.IsOpen)
                return;
            audioFrame.data = data;
            audioFrame.dataSize = legnth;
            audioFrame.bitDepth = audioDesc.bitSize;
            audioFrame.channels = audioDesc.channel;
            audioFrame.sampleRate = audioDesc.sampleRate;
            audioFrame.timestamp = DateTime.Now.Ticks / 10000 - openTime;
            LogHelper.LogMessage("audio time:" + audioFrame.timestamp);
            mediaOutput.PushAudioFrame(ref audioFrame);
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(textBox1.Text))
                return;
            if (mediaOutput.IsOpen)
                mediaOutput.Close();
            //音频与视频解码所需要信息
            OeipAudioEncoder audioEncoder = new OeipAudioEncoder();
            audioEncoder.bitrate = 12800;
            audioEncoder.channel = audioDesc.channel;
            audioEncoder.frequency = audioDesc.sampleRate;
            mediaOutput.SetAudioEncoder(audioEncoder);
            OeipVideoEncoder videoEncoder = new OeipVideoEncoder();
            videoEncoder.bitrate = 4000000;
            videoEncoder.fps = cameraControl1.Format.fps;
            videoEncoder.width = cameraControl1.Format.width;
            videoEncoder.height = cameraControl1.Format.height;
            videoEncoder.yuvType = cameraControl1.VideoPipe.YUVFMT;
            mediaOutput.SetVideoEncoder(videoEncoder);
            //重新洗白文件
            using (var file = File.Open(textBox1.Text, FileMode.Create))
            {
                LogHelper.LogMessage("create file:" + textBox1.Text + "");
            }
            if (this.checkBox1.Checked || this.checkBox2.Checked)
            {
                OeipHelper.setAudioOutputAction(onAudioDataDelegate, null);
                OeipHelper.startAudioOutput(this.checkBox1.Checked, this.checkBox2.Checked, audioDesc);
            }
            openTime = DateTime.Now.Ticks / 10000;
            mediaOutput.Open(this.textBox1.Text, this.checkBox3.Checked, this.checkBox1.Checked || this.checkBox2.Checked);
            if (!mediaOutput.IsOpen)
            {
                LogHelper.LogMessage("file:" + this.textBox1.Text + " not open");
            }
        }

        private void btn_close_Click(object sender, EventArgs e)
        {
            if (mediaOutput.IsOpen)
                mediaOutput.Close();
            OeipHelper.closeAudioOutput();
        }

        private void MediaOutputForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (mediaOutput.IsOpen)
                mediaOutput.Close();
            OeipHelper.closeAudioOutput();
        }
    }
}
