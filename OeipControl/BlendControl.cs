using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OeipWrapper.FixPipe;
using OeipWrapper;

namespace OeipControl
{
    public partial class BlendControl : UserControl
    {
        public BlendViewPipe BlendPipe { get; private set; } = null;
        public BlendControl()
        {
            InitializeComponent();
        }

        public void NativeLoad(OeipGpgpuType gpuType, VideoFormat obj)
        {
            var pipe = OeipManager.Instance.CreatePipe<OeipPipe>(gpuType);
            BlendPipe = new BlendViewPipe(pipe);
            this.displayDx111.NativeLoad(BlendPipe, obj);
        }
    }
}
