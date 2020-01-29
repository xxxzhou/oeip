using OeipWrapper;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LivePushPipe : MonoBehaviour
{
    public OeipPipe3D pipe = null;
    public OeipYUVFMT yUVFMT = OeipYUVFMT.OEIP_YUVFMT_YUV420P;
    public int InputIndex { get; private set; } = 0;
    public int OutputIndex { get; private set; } = 0;
    private bool bPush = false;
    public Texture2D texture;
    private RenderTexture renderTexture;
    // Start is called before the first frame update
    void Awake()
    {
        pipe = OeipManager.Instance.CreatePipe<OeipPipe3D>(OeipGpgpuType.OEIP_CUDA);
        InputIndex = pipe.AddLayer("input", OeipLayerType.OEIP_INPUT_LAYER);
        int mapIndex = pipe.AddLayer("map", OeipLayerType.OEIP_MAPCHANNEL_LAYER);
        int r2y = pipe.AddLayer("rgba2yuv", OeipLayerType.OEIP_RGBA2YUV_LAYER);
        OutputIndex = pipe.AddLayer("output", OeipLayerType.OEIP_OUTPUT_LAYER);

        InputParamet input = new InputParamet();
        input.bCpu = 0;
        input.bGpu = 1;
        pipe.UpdateParamet(InputIndex, input);
        if (pipe.GpgpuType == OeipGpgpuType.OEIP_CUDA)
        {
            MapChannelParamet mp = new MapChannelParamet();
            mp.blue = 0;
            mp.green = 1;
            mp.red = 2;
            mp.alpha = 3;
            pipe.UpdateParamet(mapIndex, mp);
        }
        RGBA2YUVParamet rp = new RGBA2YUVParamet();
        rp.yuvType = yUVFMT;
        pipe.UpdateParamet(r2y, rp);
        OutputParamet op = new OutputParamet();
        op.bCpu = 1;
        op.bGpu = 0;
        pipe.UpdateParamet(OutputIndex, op);
    }

    void Start()
    {
    }

    public void SetRenderTarget(RenderTexture renderTexture)
    {
        this.renderTexture = renderTexture;
        texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.BGRA32, false);
        Graphics.CopyTexture(renderTexture, texture);
        pipe.SetInput(InputIndex, renderTexture.width, renderTexture.height, OeipDataType.OEIP_CU8C3);
        pipe.SetPipeInputGpuTex(InputIndex, texture.GetNativeTexturePtr());
    }

    public void SetPush(bool bPush)
    {
        this.bPush = bPush;
    }

    // Update is called once per frame
    void Update()
    {
        if (bPush)
        {
            Graphics.CopyTexture(renderTexture, texture);
            pipe.UpdateInTex();
            pipe.RunPipe();
        }
    }
}
