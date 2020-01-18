using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Serialization;

namespace OeipCommon
{
    public static class SettingHelper
    {
        public static void SaveSetting<T>(this T seting, string path) where T : IXmlSerializable, new()
        {
            try
            {
                using (var write = new StreamWriter(path, false, Encoding.UTF8))
                {
                    XmlSerializer serializer = new XmlSerializer(typeof(T));
                    serializer.Serialize(write, seting);
                }
            }
            catch (Exception e)
            {
                LogHelper.LogMessageEx("save xml setting error.", e);
            }
        }

        public static T ReadSetting<T>(string path) where T : IXmlSerializable, new()
        {
            T setting = new T();
            try
            {
                if (!File.Exists(path))
                {
                    var file = File.Create(path);
                    file.Close();
                    setting.SaveSetting(path);
                }
                using (FileStream stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite))
                {

                    XmlSerializer serializer = new XmlSerializer(typeof(T));
                    setting = (T)serializer.Deserialize(stream);
                }
            }
            catch (Exception e)
            {
                LogHelper.LogMessageEx("read xml setting error.", e);
            }
            return setting;
        }

        public static void ReadElement<T>(this XmlReader reader, string name, ref T t)
        {
            reader.ReadStartElement(name);
            var xmlSerial = new XmlSerializer(typeof(T));
            t = (T)xmlSerial.Deserialize(reader);
            reader.ReadEndElement();
        }

        public static void WriteElement<T>(this XmlWriter writer, string name, T t)
        {
            writer.WriteStartElement(name);
            var xmlSerial = new XmlSerializer(typeof(T));
            xmlSerial.Serialize(writer, t);
            writer.WriteEndElement();
        }
    }
}
