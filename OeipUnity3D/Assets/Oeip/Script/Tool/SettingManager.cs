using OeipCommon;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SettingManager : MSingleton<SettingManager>
{
    private string path = "Setting.xml";
    public Setting Setting { get; private set; }
    protected override void Init()
    {
        Setting = SettingHelper.ReadSetting<Setting>(path);
    }

    public override void Close()
    {
        Setting.SaveSetting(path);
    }
}
