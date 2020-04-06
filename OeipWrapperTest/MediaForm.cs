using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OeipWrapper;
using OeipWrapper.FixPipe;

namespace OeipWrapperTest
{
    public partial class MediaForm : Form
    {
        private OeipMediaPlay mediaPlay = null;
        public OeipLivePipe mediaPipe { get; private set; } = null;
        public MediaForm()
        {
            InitializeComponent();
        }
        private void MediaForm_Load(object sender, EventArgs e)
        {
            mediaPlay = OeipManager.Instance.GetMediaPlay();
            mediaPlay.OnOpenEvent += MediaPlay_OnOpenEvent;
            mediaPlay.OnVideoFrameEvent += MediaPlay_OnVideoFrameEvent;
            var pipe = OeipManager.Instance.CreatePipe<OeipPipe>(OeipGpgpuType.OEIP_DX11);
            mediaPipe = new OeipLivePipe(pipe);
            mediaPipe.OnLiveImageChange += MediaPipe_OnLiveImageChange;
        }

        private void MediaPlay_OnOpenEvent(bool arg1, bool arg2)
        {

        }

        private void MediaPlay_OnVideoFrameEvent(OeipVideoFrame videoFrame)
        {
            mediaPipe.RunLivePipe(ref videoFrame);
        }

        private void MediaPipe_OnLiveImageChange(VideoFormat obj)
        {
            Action action = () => displayDx111.NativeLoad(mediaPipe, obj);
            BeginInvoke(action);
        }

        private void btn_openSrcFile_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrEmpty(this.textBox1.Text))
            {
                var result = openFileDialog1.ShowDialog();
                if (result != DialogResult.OK && result != DialogResult.Yes)
                    return;
                textBox1.Text = openFileDialog1.FileName;
            }           
            if (mediaPlay.IsOpen)
            {
                mediaPlay.Close();
            }            
            mediaPlay.Open(textBox1.Text, true);
        }
    }
}
