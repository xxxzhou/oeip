using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OeipCommon.OeipAttribute;

namespace OeipControl.Controls
{
    public partial class ToggleControl : BaseControl<bool>
    {
        private ToggleAttribute attribute;
        public ToggleControl()
        {
            InitializeComponent();
            this.label = this.label1;
        }

        public override void OnSetAttribute()
        {
            attribute = Attribute<ToggleAttribute>();
            OnSetValue(attribute.Default);
        }

        public override void OnSetValue(bool value)
        {
            if (attribute == null)
                return;
            this.checkBox1.Checked = value;
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            OnValueChange(this.checkBox1.Checked);
        }
    }
}
