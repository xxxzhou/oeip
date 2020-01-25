using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.FixPipe
{
    /// <summary>
    /// 用于把二个GPU纹理合成一个展示
    /// </summary>
    public class BlendViewPipe : IDXViewPipe
    {
        public OeipPipe Pipe { get; private set; }

        public bool IsGpu { get; private set; } = true;

        public int OutGpuIndex { get; private set; } = 3;
        public int OutYuvIndex { get; private set; } = 0;

        public int InputMain { get; private set; } = 0;
        public int InputAux { get; private set; } = 0;

        public int BlendIndex { get; private set; } = 0;

        private BlendParamet blendParamet = new BlendParamet();

        public OeipYUVFMT YUVFMT { get; private set; } = OeipYUVFMT.OEIP_YUVFMT_YUV420P;

        public ref BlendParamet Blend
        {
            get
            {
                return ref blendParamet;
            }
        }

        public BlendViewPipe(OeipPipe pipe)
        {
            this.Pipe = pipe;

            InputMain = pipe.AddLayer("input main", OeipLayerType.OEIP_INPUT_LAYER);
            InputAux = pipe.AddLayer("input aux", OeipLayerType.OEIP_INPUT_LAYER);
            BlendIndex = pipe.AddLayer("blend", OeipLayerType.OEIP_BLEND_LAYER);
            OutGpuIndex = pipe.AddLayer("out tex", OeipLayerType.OEIP_OUTPUT_LAYER);
            int rgb2Yuv = pipe.AddLayer("rgba2yuv", OeipLayerType.OEIP_RGBA2YUV_LAYER);
            OutYuvIndex = pipe.AddLayer("out yuv tex", OeipLayerType.OEIP_OUTPUT_LAYER);

            pipe.ConnectLayer(BlendIndex, InputMain, 0, 0);
            pipe.ConnectLayer(BlendIndex, InputAux, 0, 1);
            //混合显示的位置
            blendParamet.rect.centerX = 0.7f;
            blendParamet.rect.centerY = 0.7f;
            blendParamet.rect.width = 0.5f;
            blendParamet.rect.height = 0.5f;
            blendParamet.opacity = 0.0f;
            pipe.UpdateParamet(BlendIndex, blendParamet);
            //要求输入的是显存数据
            InputParamet inParamet = new InputParamet();
            inParamet.bGpu = 1;
            inParamet.bCpu = 0;
            pipe.UpdateParamet(InputMain, inParamet);
            pipe.UpdateParamet(InputAux, inParamet);

            RGBA2YUVParamet yuvParamet = new RGBA2YUVParamet();
            yuvParamet.yuvType = YUVFMT;
            Pipe.UpdateParamet(rgb2Yuv, yuvParamet);
        }
    }
}
