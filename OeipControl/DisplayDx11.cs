using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using SharpDX.Windows;
using SharpDX.DXGI;
using SharpDX.Direct3D11;
using SharpDX.Direct3D;
using SharpDX.Mathematics.Interop;
using OeipWrapper;
using OeipWrapper.FixPipe;

namespace OeipControl
{
    public partial class DisplayDx11 : UserControl
    {
        private SharpDX.Direct3D11.Device deviceDx11 = null;
        private DeviceContext deviceCtx = null;
        private SwapChain swapChain = null;
        private RenderTargetView renderTargetView = null;
        private Texture2D backBuffer = null;
        private ISharpDXViewPipe viewPipe = null;
        public int TexWidth { get; private set; } = 1920;
        public int TexHeight { get; private set; } = 1080;

        public DisplayDx11()
        {
            InitializeComponent();
            this.timer.Tick += Timer_Tick;
        }

        private void Timer_Tick(object sender, EventArgs e)
        {
            if (!viewPipe.IsGpu)
                return;
            this.Draw();
        }

        public void NativeLoad(ISharpDXViewPipe videoPipe, VideoFormat videoFormat)
        {
            this.timer.Enabled = false;
            ModeDescription backBufferDesc = new ModeDescription(videoFormat.width, videoFormat.height, new Rational(videoFormat.fps, 1), Format.R8G8B8A8_UNorm);
            SwapChainDescription swapChainDesc = new SwapChainDescription()
            {
                ModeDescription = backBufferDesc,
                SampleDescription = new SampleDescription(1, 0),
                Usage = Usage.RenderTargetOutput,
                BufferCount = 1,
                OutputHandle = this.Handle,
                IsWindowed = true
            };
            SharpDX.Direct3D11.Device.CreateWithSwapChain(DriverType.Hardware, DeviceCreationFlags.None, swapChainDesc,
                out deviceDx11, out swapChain);
            deviceCtx = deviceDx11.ImmediateContext;
            backBuffer = swapChain.GetBackBuffer<Texture2D>(0);
            renderTargetView = new RenderTargetView(deviceDx11, backBuffer);
            deviceCtx.OutputMerger.SetRenderTargets(renderTargetView);
            viewPipe = videoPipe;
            this.timer.Interval = 1000 / videoFormat.fps;
            this.timer.Enabled = true;
            //Action action = () => { RenderLoop.Run(this, Draw); };
            //this.BeginInvoke(action);
        }

        private void Draw()
        {
            viewPipe.Pipe.setPipeOutputGpuTex(viewPipe.OutGpuIndex, deviceDx11.NativePointer, backBuffer.NativePointer);
            swapChain.Present(1, PresentFlags.None);
        }
    }
}
