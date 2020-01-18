using OeipWrapper.FixPipe;
using OeipCommon;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;
using System.Xml.Schema;
using System.Xml.Serialization;

namespace OeipWrapperTest
{
    public class Setting : IXmlSerializable
    {
        public OeipVideoParamet videoParamet = new OeipVideoParamet();
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

            reader.ReadEndElement();
            reader.ReadEndElement();
        }

        public void WriteXml(XmlWriter writer)
        {
            writer.WriteStartElement("XmlSerializable");

            writer.WriteElement("VideoParamet", videoParamet);

            writer.WriteEndElement();
        }
    }
}
