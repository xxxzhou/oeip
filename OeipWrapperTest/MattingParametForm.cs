using OeipControl.Controls;
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
    public partial class MattingParametForm : Form
    {
        private Setting setting = null;
        public MattingParametForm()
        {
            InitializeComponent();
        }

        private void MattingParametForm_Load(object sender, EventArgs e)
        {
            setting = SettingManager.Instance.Setting;
        }

        public void SetBindObj(ObjectBindWF<OeipVideoParamet> objectAttribute)
        {
            objectAttribute.Bind(setting.videoParamet, flowLayoutPanel1);
        }
    }
}
