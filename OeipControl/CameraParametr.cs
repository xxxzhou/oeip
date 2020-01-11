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
    public partial class CameraParametr : UserControl
    {
        private CamParametrs camParametrs;
        private int cameraId = 0;
        public string[] names = new string[17];

        public CameraParametr()
        {
            InitializeComponent();

            names[0] = "亮度";
            names[1] = "对比度";
            names[2] = "色调";
            names[3] = "饱和度";
            names[4] = "清晰度";
            names[5] = "伽玛";
            names[6] = "启用颜色";
            names[7] = "白平衡";
            names[8] = "逆光对比";
            names[9] = "增益";

            names[10] = "光圈";
            names[11] = "倾斜";
            names[12] = "滚动";
            names[13] = "缩放";
            names[14] = "曝光";
            names[15] = "全景";
            names[16] = "焦点";
        }

        public void SetCameraParamet(ref CamParametrs parametrs, int id)
        {
            camParametrs = parametrs;
            cameraId = id;
            LoadParametrs();
        }

        private unsafe void LoadParametrs()
        {
            fixed (Parametr* par = &camParametrs.Brightness)
            {
                for (int i = 0; i < names.Length; i++)
                {
                    FlowLayoutPanel flowLayoutPanel = i < 10 ? flowLayoutPanel1 : flowLayoutPanel2;
                    ParametrControl cameraParametrControl = new ParametrControl();
                    cameraParametrControl.InitControl(ref *(par + i), names[i], i, i == 7 || i == 14 || i == 16);
                    cameraParametrControl.OnParametrEvent += CameraParametrControl_OnParametrEvent;
                    flowLayoutPanel.Controls.Add(cameraParametrControl);
                }
            }
        }

        private unsafe void CameraParametrControl_OnParametrEvent(Parametr parametr, int index)
        {
            fixed (Parametr* par = &camParametrs.Brightness)
            {
                *(par + index) = parametr;
            }
            OeipHelper.setDeviceParametrs(cameraId, ref camParametrs);
        }
    }
}
