using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace DataAnalysis
{
    [Serializable]
    public class DataPath
    {
        //标签JSON文件存放
        public string AnnotationPath { get; set; } = string.Empty;
        //原始Image文件存放
        public string SourceImagePath { get; set; } = string.Empty;
        //生成的label文件存放
        public string LabelPath { get; set; } = string.Empty;
        //根据选择复制文件的Image存放路径
        public string DestImagePath { get; set; } = string.Empty;
    }

    [Serializable]
    public class COCOData
    {
        public DataPath TrainData { get; set; } = new DataPath();
        public DataPath PredData { get; set; } = new DataPath();
    }

    public class Setting
    {
        public COCOData Data = new COCOData();

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

            ReadElement(reader, "COCOData", ref Data);

            reader.ReadEndElement();
            reader.ReadEndElement();
        }

        public void WriteXml(XmlWriter writer)
        {           
            writer.WriteStartElement("XmlSerializable");
            WriteElement(writer, "COCOData", Data);
            writer.WriteEndElement();
        }

        public void ReadElement<T>(XmlReader reader, string name, ref T t)
        {
            reader.ReadStartElement(name);
            var xmlSerial = new XmlSerializer(typeof(T));
            t = (T)xmlSerial.Deserialize(reader);
            reader.ReadEndElement();
        }

        public void WriteElement<T>(XmlWriter writer, string name, T t)
        {
            writer.WriteStartElement(name);
            var xmlSerial = new XmlSerializer(typeof(T));
            xmlSerial.Serialize(writer, t);
            writer.WriteEndElement();
        }
    }
}
