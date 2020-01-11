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

namespace ZmfWrapperWF
{
    public partial class ParametrControl : UserControl
    {
        public delegate void onParametrAction(Parametr parametr, int index);
        public event onParametrAction OnParametrEvent;
        private int index = 0;
        public Parametr parametr;

        public ParametrControl()
        {
            InitializeComponent();

            this.trackBar1.ValueChanged += TrackBar1_ValueChanged;
            this.textBox1.TextChanged += TextBox1_TextChanged;
            this.checkBox1.CheckedChanged += CheckBox1_CheckedChanged;
        }

        private void CheckBox1_CheckedChanged(object sender, EventArgs e)
        {
            parametr.Flag = this.checkBox1.Checked ? 1 : 2;
            this.trackBar1.Enabled = !this.checkBox1.Checked;
            this.textBox1.Enabled = !this.checkBox1.Checked;
            OnParametrEvent?.Invoke(parametr, index);
        }

        private void TextBox1_TextChanged(object sender, EventArgs e)
        {
            if (int.TryParse(this.textBox1.Text, out int value))
            {
                if (value != this.trackBar1.Value)
                {
                    if (value >= parametr.Min && value <= parametr.Max)
                        this.trackBar1.Value = value;
                }
            }
        }

        private void TrackBar1_ValueChanged(object sender, EventArgs e)
        {
            if (this.textBox1.Text != this.trackBar1.Value.ToString())
            {
                this.textBox1.Text = this.trackBar1.Value.ToString();
            }
            if (parametr.CurrentValue != this.trackBar1.Value)
            {
                parametr.CurrentValue = this.trackBar1.Value;
                OnParametrEvent?.Invoke(parametr, index);
            }
        }

        //typedef enum tagVideoProcAmpFlags
        //{
        //    VideoProcAmp_Flags_Auto = 0x0001,
        //    VideoProcAmp_Flags_Manual = 0x0002,
        //}
        //VideoProcAmpFlags;
        public void InitControl(ref Parametr par, string text, int pindex, bool bCheck = false)
        {
            if (par.Flag == 0)
            {
                this.trackBar1.Enabled = false;
                this.textBox1.Enabled = false;
                this.checkBox1.Enabled = false;
            }
            else
            {
                this.checkBox1.Checked = par.Flag == 1 ? true : false;
                this.checkBox1.Enabled = bCheck;
            }
            this.label1.Text = text;
            this.trackBar1.Minimum = par.Min;
            this.trackBar1.Maximum = par.Max;

            parametr = par;
            this.textBox1.Text = parametr.CurrentValue.ToString();
            this.trackBar1.Value = parametr.CurrentValue;
            index = pindex;
        }
    }
}
