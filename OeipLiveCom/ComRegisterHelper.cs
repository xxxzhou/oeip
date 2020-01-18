using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipLiveCom
{
    public class ComRegisterHelper
    {
        const int OLEMISC_RECOMPOSEONRESIZE = 1;
        const int OLEMISC_CANTLINKINSIDE = 16;
        const int OLEMISC_INSIDEOUT = 128;
        const int OLEMISC_ACTIVATEWHENVISIBLE = 256;
        const int OLEMISC_SETCLIENTSITEFIRST = 131072;

        public static void Register(Type t)
        {
            string keyName = @"CLSID\" + t.GUID.ToString("B");
            using (RegistryKey key = Registry.ClassesRoot.OpenSubKey(keyName, true))
            {
                key.CreateSubKey("Control").Close();
                using (RegistryKey subkey = key.CreateSubKey("MiscStatus"))
                {
                    // 131456 decimal == 0x20180.
                    long val = (long)(OLEMISC_INSIDEOUT | OLEMISC_ACTIVATEWHENVISIBLE | OLEMISC_SETCLIENTSITEFIRST);
                    subkey.SetValue("", val);
                }
                using (RegistryKey subkey = key.CreateSubKey("TypeLib"))
                {
                    Guid libid = Marshal.GetTypeLibGuidForAssembly(t.Assembly);
                    subkey.SetValue("", libid.ToString("B"));
                }
                using (RegistryKey subkey = key.CreateSubKey("Version"))
                {
                    Version ver = t.Assembly.GetName().Version;
                    string version = string.Format("{0}.{1}", ver.Major, ver.Minor);
                    subkey.SetValue("", version);
                }
                // Next create the CodeBase entry - needed if not string named and GACced.
                using (RegistryKey subkey = key.OpenSubKey("InprocServer32", true))
                {
                    subkey.SetValue("CodeBase", Assembly.GetExecutingAssembly().CodeBase);
                }
                // Finally close the main key
            }
        }

        public static void Unregister(Type t)
        {
            // Delete the entire CLSID\{clsid} subtree for this component.     
            //if (Environment.Is64BitOperatingSystem)
            //string keyName = @"Wow6432Node\CLSID\" + t.GUID.ToString("B");
            string keyName = @"CLSID\" + t.GUID.ToString("B");
            Registry.ClassesRoot.DeleteSubKeyTree(keyName);
        }
    }
}
