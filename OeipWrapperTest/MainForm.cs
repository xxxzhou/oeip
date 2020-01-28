using OeipCommon;
using OeipCommon.OeipAttribute;
using OeipControl.Controls;
using OeipWrapper;
using OeipWrapper.FixPipe;
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
    public partial class MainForm : Form
    {
        private ObjectBindWF<OeipVideoParamet> objectAttribute;
        private Setting setting = null;
        public MainForm()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            setting = SettingManager.Instance.Setting;
            OeipManager.Instance.OnLogEvent += Instance_OnLogEvent;
            this.cameraControl1.NativeLoad(OeipGpgpuType.OEIP_CUDA, 0, false);
            objectAttribute = new ObjectBindWF<OeipVideoParamet>();
            objectAttribute.OnChangeEvent += ObjectAttribute_OnChangeEvent;
            this.cameraControl1.VideoPipe.UpdateVideoParamet(setting.videoParamet);
        }

        private void ObjectAttribute_OnChangeEvent(ObjectBind<OeipVideoParamet> arg1, string arg2)
        {
            this.cameraControl1.VideoPipe.UpdateVideoParamet(arg1.Obj);
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

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            OeipManager.Instance.Close();
            SettingManager.Instance.Close();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            MattingParametForm mattingParametForm = new MattingParametForm();
            mattingParametForm.Show();
            mattingParametForm.SetBindObj(objectAttribute);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            OeipManager.Instance.Close();
        }
    }
}
