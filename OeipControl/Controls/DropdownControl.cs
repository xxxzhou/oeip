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
    public partial class DropdownControl : BaseControl<int>
    {
        private DropdownAttribute attribute = null;
        internal DropdownControl parent = null;
        private Func<List<string>> onFillFunc;
        private Func<int, List<string>> onFillParentFunc;
        private bool bAutoAddDefault = false;

        public DropdownControl()
        {
            InitializeComponent();
            this.label = this.label1;
        }

        public override void OnSetAttribute()
        {
            attribute = Attribute<DropdownAttribute>();
            bAutoAddDefault = attribute.IsAutoDefault;
            OnSetValue(attribute.Default);
        }

        public override void OnSetValue(int value)
        {
            if (attribute == null)
                return;
            int index = value;
            if (bAutoAddDefault)
            {
                index = index + 1;
            }
            if (index >= 0 && index < comboBox1.Items.Count)
            {
                comboBox1.SelectedIndex = index;
            }
        }

        private void fillOption()
        {
            comboBox1.Items.Clear();
            List<string> options = new List<string>();
            if (bAutoAddDefault)
            {
                options.Add("None");
            }
            options.AddRange(onFillFunc());
            foreach (var option in options)
            {
                comboBox1.Items.Add(option);
            }
        }

        private void parent_CheckedChanged(object sender, EventArgs e)
        {
            int value = (sender as ComboBox).SelectedIndex;
            comboBox1.Items.Clear();
            List<string> options = new List<string>();
            if (bAutoAddDefault)
            {
                options.Add("None");
            }
            int index = value;
            if (parent.bAutoAddDefault)
                index = value - 1;
            options.AddRange(onFillParentFunc(index));
            foreach (var option in options)
            {
                comboBox1.Items.Add(option);
            }
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            int index = comboBox1.SelectedIndex;
            if (bAutoAddDefault)
            {
                index = index - 1;
            }
            OnValueChange(index);
        }

        public void SetFillOptions(Func<List<string>> onFunc)
        {
            onFillFunc = onFunc;
            fillOption();
        }

        public void SetFillOptions(Func<int, List<string>> onFunc)
        {
            onFillParentFunc = onFunc;
            if (parent != null)
                parent.comboBox1.SelectedIndexChanged += parent_CheckedChanged;
        }

    }
}
