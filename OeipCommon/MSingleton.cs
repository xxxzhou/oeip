using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace OeipCommon
{
    public abstract class MSingleton<T> where T : MSingleton<T>, new()
    {
        protected static T instance = null;

        protected MSingleton()
        {
        }

        public static T Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new T();
                    instance.Init();
                }
                return instance;
            }
        }

        protected abstract void Init();

        public virtual void Close() { }
    }

    public class SettingSingleton<T> : MSingleton<SettingSingleton<T>> where T : IXmlSerializable, new()
    {
        public T Setting = new T();

        protected override void Init()
        {

        }

        public void SaveSetting(string path)
        {
            using (var write = new StreamWriter(path, false, Encoding.UTF8))
            {
                XmlSerializer serializer = new XmlSerializer(typeof(T));
                serializer.Serialize(write, Setting);
            }
        }

        public void ReadSetting(string path)
        {
            if (!File.Exists(path))
            {
                var file = File.Create(path);
                file.Close();
                SaveSetting(path);
            }
            using (FileStream stream = new FileStream(path, FileMode.OpenOrCreate, FileAccess.ReadWrite))
            {
                try
                {
                    XmlSerializer serializer = new XmlSerializer(typeof(T));
                    Setting = (T)serializer.Deserialize(stream);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.Message);
                }
            }
        }
    }
}
