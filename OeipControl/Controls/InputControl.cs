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
    public partial class InputControl : BaseControl<string>
    {
        private InputAttribute attribute;

        public InputControl()
        {
            InitializeComponent();
            this.label1 = this.label;
        }

        public override void OnSetAttribute()
        {
            attribute = Attribute<InputAttribute>();
            OnSetValue(attribute.Default);
        }

        public override void OnSetValue(string value)
        {
            if (attribute == null)
                return;
            this.textBox1.Text = value;
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            OnValueChange(this.textBox1.Text);
        }
    }
}
