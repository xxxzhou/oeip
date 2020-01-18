using System;
using OeipCommon.OeipAttribute;

namespace OeipControl.Controls
{
    public partial class SliderInputControl : BaseControl<float>
    {
        //如果是浮点数
        private float ScaleRange = 1.0f;
        private int Range = 1;
        private bool bInt = false;
        private SliderInputAttribute attribute;

        private bool bSliderChange = false;
        private bool bTextChange = false;

        public SliderInputControl()
        {
            InitializeComponent();
            this.label = this.label1;
        }

        public override void OnSetAttribute()
        {
            attribute = Attribute<SliderInputAttribute>();
            bInt = attribute.IsInt;
            ScaleRange = bInt ? 1.0f : 100.0f;
            if (attribute.IsAutoRange)
                Range = (int)(attribute.Range * ScaleRange);
            else
                Range = (int)((attribute.Max - attribute.Min) * ScaleRange);
            trackBar1.Minimum = (int)(attribute.Min * ScaleRange);
            trackBar1.Maximum = (int)(attribute.Max * ScaleRange);
            setValue(attribute.Default * ScaleRange);
        }

        private void setValue(float value)
        {
            if (attribute == null)
                return;
            int svalue = (int)(value);
            if (svalue < trackBar1.Minimum || svalue > trackBar1.Maximum)
                return;
            if (attribute.IsAutoRange)
            {
                this.trackBar1.Minimum = svalue - Range / 2;
                this.trackBar1.Maximum = svalue + Range / 2;
            }
            this.trackBar1.Value = svalue;
            this.textBox1.Text = (value / ScaleRange).ToString();
        }

        public override void OnSetValue(float value)
        {
            if (attribute == null)
                return;
            setValue(value);
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            if (bTextChange)
                return;
            bTextChange = true;
            if (float.TryParse(this.textBox1.Text, out float fvalue))
            {
                setValue(fvalue * ScaleRange);
            }
            bTextChange = false;
        }

        private void trackBar1_ValueChanged(object sender, EventArgs e)
        {
            if (bSliderChange)
                return;
            bSliderChange = true;
            int value = this.trackBar1.Value;
            float svalue = value / ScaleRange;
            setValue(value);
            OnValueChange(svalue);
            bSliderChange = false;
        }
    }
}
