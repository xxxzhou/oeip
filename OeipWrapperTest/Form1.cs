using OeipCommon;
using OeipWrapper;
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
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            OeipManager.Instance.OnLogEvent += Instance_OnLogEvent;
            this.cameraControl1.NativeLoad(OeipGpgpuType.OEIP_CUDA, 1, false);
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
        }
    }
}
