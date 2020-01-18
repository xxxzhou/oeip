using OeipCommon;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapperTest
{
    public class SettingManager : MSingleton<SettingManager>
    {
        private string path = "OeipWrapperTestSetting.xml";
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
}
