using OeipWrapper;
using OeipWrapper.FixPipe;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraView : MonoBehaviour
{
    [SerializeField]
    public Texture2D SourceTex = null;
    private OeipCamera oeipCamera = null;
    private OeipPipe3D cameraPipe = null;
    public OeipVideoPipe VideoPipe = null;

    private int formatIndex = 0;
    private VideoFormat selectFormat;

    private void Awake()
    {
        //创建一个管线
        cameraPipe = OeipManager.Instance.CreatePipe<OeipPipe3D>(OeipGpgpuType.OEIP_CUDA);
        //生成一个视频渲染管线
        VideoPipe = new OeipVideoPipe(cameraPipe);
        VideoPipe.SetOutput(false, true);

        //生成一个摄像机处理类
        oeipCamera = new OeipCamera();
        oeipCamera.OnReviceEvent += OeipCamera_OnReviceEvent;
    }

    private void OeipCamera_OnReviceEvent(IntPtr data, int width, int height)
    {
        //摄像机输入数据
        VideoPipe.RunVideoPipe(data);
    }

    void Update()
    {
        if (cameraPipe == null)
            return;
        cameraPipe.UpdateOutTex();
    }

    //打开摄像机
    public void OpenCamera(int cameraIndex, int formatIndex)
    {
        if (cameraIndex < 0 || cameraIndex >= OeipManager.Instance.OeipDevices.Count)
        {
            return;
        }
        if (oeipCamera != null)
        {
            oeipCamera.Close();
        }
        oeipCamera.SetDevice(OeipManager.Instance.OeipDevices[cameraIndex]);
        this.formatIndex = formatIndex;
        if (this.formatIndex < 0)
            this.formatIndex = oeipCamera.FindFormatIndex(1920, 1080);
        SetFormat(this.formatIndex);
    }

    /// <summary>
    /// 设定摄像机的输出格式
    /// </summary>
    /// <param name="index"></param>
    private void SetFormat(int index)
    {
        selectFormat = oeipCamera.VideoFormats[index];
        //设定处理图像格式,图像宽度，图像长度
        VideoPipe.SetVideoFormat(selectFormat.videoType, selectFormat.width, selectFormat.height);

        oeipCamera.SetFormat(index);
        oeipCamera.Open();

        if (SourceTex == null || SourceTex.width != selectFormat.width || SourceTex.height != selectFormat.height)
        {
            SourceTex = new Texture2D(selectFormat.width, selectFormat.height, TextureFormat.RGBA32, false);
            SourceTex.Apply();
            //Unity 纹理作为输出
            cameraPipe.SetPipeOutputGpuTex(VideoPipe.OutGpuIndex, SourceTex.GetNativeTexturePtr());
        }
    }
}
