using OeipCommon;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace DataAnalysis
{
    public class SettingManager : MSingleton<SettingManager>
    {
        public Setting Setting = new Setting();

        private string path = "DASetting.xml";// Application.dataPath + "/Resources/Xml/" + "Setting.xml";

        //public string bgPath = Application.ExecutablePath + "/BG";

        protected override void Init()
        {
            if (!File.Exists(path))
            {
                var file = File.Create(path);
                file.Close();
                defaultWriteXML();
            }
            using (FileStream stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite))
            {
                try
                {
                    XmlSerializer serializer = new XmlSerializer(typeof(Setting));
                    Setting = (Setting)serializer.Deserialize(stream);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
        }

        public void defaultWriteXML()
        {
            this.Setting = new Setting();
            SaveSetting();
        }

        public void SaveSetting()
        {
            //StreamWriter FileWriter = new StreamWriter(path, false, Encoding.UTF8);
            using (var write = new StreamWriter(path, false, Encoding.UTF8))
            {
                XmlSerializer serializer = new XmlSerializer(typeof(Setting));
                serializer.Serialize(write, Setting);
            }
        }

        public override void Close()
        {
            SaveSetting();
        }
    }
}
