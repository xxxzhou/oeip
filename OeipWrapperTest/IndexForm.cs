using OeipCommon;
using OeipWrapper;
using OeipWrapper.Live;
using System;
using System.Windows.Forms;

namespace OeipWrapperTest
{
    public partial class IndexForm : Form
    {
        public IndexForm()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            MainForm mainForm = new MainForm();
            mainForm.Show();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            LiveForm liveForm = new LiveForm();
            liveForm.Show();
        }

        private void IndexForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            //OeipLiveManager.Instance.Close();
            OeipManager.Instance.Close();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            MediaForm mediaForm = new MediaForm();
            mediaForm.Show();
        }

        private void button4_Click(object sender, EventArgs e)
        {
            MediaOutputForm mediaOutputForm = new MediaOutputForm();
            mediaOutputForm.Show();
        }
    }
}
