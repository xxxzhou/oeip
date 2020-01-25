using OeipWrapper;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OeipPipe3D : OeipPipe
{
    public void SetPipeInputGpuTex(int layerIndex, IntPtr tex, int inputIndex = 0)
    {
        OeipHelperUnity3D.SetPipeInputGpuTex(PipeId, layerIndex, tex, inputIndex);
        bSetInput = true;
    }

    public void SetPipeOutputGpuTex(int layerIndex, IntPtr tex, int outputIndex = 0)
    {
        OeipHelperUnity3D.SetPipeOutputGpuTex(PipeId, layerIndex, tex, outputIndex);
    }

    public void UpdateInTex()
    {
        GL.IssuePluginEvent(OeipHelperUnity3D.SetUpdateTexFunc(), PipeId);
    }

    /// <summary>
    /// 更新需要
    /// </summary>
    /// <param name="pipeId"></param>
    public void UpdateOutTex()
    {
        GL.IssuePluginEvent(OeipHelperUnity3D.GetUpdateTexFunc(), PipeId);
    }
}
