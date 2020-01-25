using OeipCommon;
using OeipCommon.OeipAttribute;
using OeipWrapper.FixPipe;
using System.Collections;
using System.Collections.Generic;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;
using UnityEngine;

public class CameraSetting
{
    [Dropdown(DisplayName = "选择摄像头", Order = 1)]
    public int CameraIndex = 0;
    [Dropdown(DisplayName = "选择输出格式", Parent = "CameraIndex", Order = 2)]
    public int FormatIndex = 0;
}

public class Setting : IXmlSerializable
{
    public OeipVideoParamet videoParamet = new OeipVideoParamet();
    public CameraSetting cameraSetting = new CameraSetting();
    public Setting()
    {
    }

    public XmlSchema GetSchema()
    {
        return null;
    }

    public void ReadXml(XmlReader reader)
    {
        reader.ReadStartElement("Setting");
        reader.ReadStartElement("XmlSerializable");

        reader.ReadElement("VideoParamet", ref videoParamet);
        reader.ReadElement("CameraSetting", ref cameraSetting);

        reader.ReadEndElement();
        reader.ReadEndElement();
    }

    public void WriteXml(XmlWriter writer)
    {
        writer.WriteStartElement("XmlSerializable");

        writer.WriteElement("VideoParamet", videoParamet);
        writer.WriteElement("CameraSetting", cameraSetting);

        writer.WriteEndElement();
    }
}
