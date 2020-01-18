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
}
