using OeipCommon;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace OeipLiveMedia
{
    class Program
    {


        static void Main(string[] args)
        {
            try
            {
                //TaskTest taskTest = new TaskTest();
                //taskTest.Test();
                LiveMedia mediaServer = new LiveMedia();
                mediaServer.Run();
                LogHelper.LogMessage("启动媒体服务器");
                Console.ReadLine();
                mediaServer.Close();
            }
            catch (Exception ex)
            {
                LogHelper.LogMessageEx("main:", ex);
            }
        }
    }
}
