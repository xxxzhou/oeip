using OeipCommon;
using OeipWrapper;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class OeipManagerU3D : MSingleton<OeipManagerU3D>
{
    public SliderInputComponent SliderInputTemplate { get; set; }
    public InputComponent InputComponent { get; set; }
    public DropdownComponent DropdownComponent { get; set; }
    public ToggleComponent ToggleComponent { get; set; }

    protected override void Init()
    {

    }

    public List<string> GetCameras()
    {
        List<string> cameras = new List<string>();
        var cameraList = OeipManager.Instance.OeipDevices;
        cameraList.ForEach(p => cameras.Add(p.ToString()));
        return cameras;
    }

    public List<string> GetFormats(int index)
    {
        List<string> formatList = new List<string>();
        List<VideoFormat> formats = OeipManager.Instance.GetCameraFormats(index);
        foreach (var format in formats)
        {
            formatList.Add(format.width + "x" + format.height + " " + format.fps + "fps " + format.GetVideoType());
        }
        return formatList;
    }
}
