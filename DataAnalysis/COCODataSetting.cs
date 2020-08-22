using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DataAnalysis
{
    public partial class COCODataSetting : Form
    {
        class COCOItem
        {
            public int index { get; set; }
            public string text { get; set; }
        }

        private DataPath selectData = null;

        private Setting setting = null;
        public COCODataSetting()
        {
            InitializeComponent();
            setting = SettingManager.Instance.Setting;
        }

        private void COCODataSetting_Load(object sender, EventArgs e)
        {
            this.comboBox1.DisplayMember = "text";
            this.comboBox1.ValueMember = "index";
            this.comboBox1.Items.Add(new COCOItem() { index = 0, text = "训练集" });
            this.comboBox1.Items.Add(new COCOItem() { index = 1, text = "测试集" });
            this.comboBox1.SelectedIndex = 0;
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            COCOItem selectItem = this.comboBox1.SelectedItem as COCOItem;
            selectData = null;
            if (selectItem.index == 0)
                selectData = setting.Data.TrainData;
            else if (selectItem.index == 1)
                selectData = setting.Data.PredData;
            if (selectData == null)
                return;
            this.textBox1.Text = selectData.AnnotationPath;
            this.textBox2.Text = selectData.SourceImagePath;
            this.textBox3.Text = selectData.LabelPath;
            this.textBox4.Text = selectData.DestImagePath;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (selectData == null)
                return;
            this.openFileDialog1.DefaultExt = "*.json";
            this.openFileDialog1.FileName = selectData.AnnotationPath;
            if (this.openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                selectData.AnnotationPath = this.openFileDialog1.FileName;
                this.textBox1.Text = selectData.AnnotationPath;
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (selectData == null)
                return;
            this.folderBrowserDialog1.SelectedPath = selectData.SourceImagePath;
            if (this.folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                selectData.SourceImagePath = this.folderBrowserDialog1.SelectedPath;
                this.textBox2.Text = selectData.SourceImagePath;
            }
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (selectData == null)
                return;
            this.folderBrowserDialog1.SelectedPath = selectData.LabelPath;
            if (this.folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                selectData.LabelPath = this.folderBrowserDialog1.SelectedPath;
                this.textBox3.Text = selectData.LabelPath;
            }
        }

        private void button4_Click(object sender, EventArgs e)
        {
            if (selectData == null)
                return;
            this.folderBrowserDialog1.SelectedPath = selectData.DestImagePath;
            if (this.folderBrowserDialog1.ShowDialog() == DialogResult.OK)
            {
                selectData.DestImagePath = this.folderBrowserDialog1.SelectedPath;
                this.textBox4.Text = selectData.DestImagePath;
            }
        }
    }
}
