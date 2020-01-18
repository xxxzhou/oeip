using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNet.SignalR;
using Microsoft.Owin.Cors;
using Microsoft.Owin.Hosting;
using OeipCommon;
using Owin;

namespace OeipLiveServer
{
    class Program
    {
        static void Main(string[] args)
        {
            LogHelper.LogMessage("启动服务器.");
            string url = RoomManager.Instance.SelfHost;
            LogHelper.LogMessage($"Server running on {url}");
            using (WebApp.Start<Startup>(new StartOptions(url)))
            {
                //创建HttpCient测试webapi 
                HttpClient client = new HttpClient();
                //通过get请求数据
                var response = client.GetAsync(url).Result;
                //打印请求结果
                Console.WriteLine(response);
                Console.ReadLine();
            }
        }
    }
}
