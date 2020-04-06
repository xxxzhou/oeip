using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Drawing.Imaging;

namespace OeipControl
{
    public partial class DisplayWF : UserControl
    {
        public int TexWidth { get; private set; } = 1920;
        public int TexHeight { get; private set; } = 1080;

        private Bitmap mainMap = new Bitmap(1920, 1080, PixelFormat.Format32bppRgb);
        private object obj = new object();
        public DisplayWF()
        {
            InitializeComponent();
        }

        public void UpdateImage(int width, int height, IntPtr data)
        {
            if (data == IntPtr.Zero || this.Disposing || this.IsDisposed || !this.IsHandleCreated)
                return;
            lock (obj)
            {
                if (mainMap == null || mainMap.Width != width || mainMap.Height != height)
                {
                    mainMap = new Bitmap(width, height, PixelFormat.Format32bppRgb);
                    TexWidth = width;
                    TexHeight = height;
                }
                var mapData = mainMap.LockBits(new Rectangle(0, 0, mainMap.Width, mainMap.Height), ImageLockMode.WriteOnly, PixelFormat.Format32bppRgb);
                unsafe
                {
                    Buffer.MemoryCopy(data.ToPointer(), mapData.Scan0.ToPointer(), mainMap.Width * mainMap.Height * 4, mainMap.Width * mainMap.Height * 4);
                }
                mainMap.UnlockBits(mapData);
            }
            Action action = () =>
            {
                pctBox.Refresh();
            };
            //invoke在关闭窗口时，会引发访问已Dispose对象异常
            this.BeginInvoke(action);
        }

        private void pictureBox1_Paint(object sender, PaintEventArgs e)
        {
            lock (obj)
                e.Graphics.DrawImage(mainMap, pctBox.ClientRectangle);
        }
    }
}
