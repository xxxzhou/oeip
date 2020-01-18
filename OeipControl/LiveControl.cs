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
    public partial class LiveControl : UserControl
    {
        public OeipLivePipe LivePipe { get; private set; } = null;

        public LiveControl()
        {
            InitializeComponent();
        }

        public void NativeLoad(OeipGpgpuType gpuType, int index = 0, bool bCpu = false)
        {
            var pipe = OeipManager.Instance.CreatePipe<OeipPipe>(gpuType);
            LivePipe = new OeipLivePipe(pipe);
            LivePipe.OnLiveImageChange += LivePipe_OnLiveImageChange;
        }

        private void LivePipe_OnLiveImageChange(VideoFormat obj)
        {
            Action action = () => displayDx111.NativeLoad(LivePipe, obj);
            BeginInvoke(action);
        }
    }
}
