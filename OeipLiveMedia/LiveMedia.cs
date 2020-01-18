using Microsoft.AspNet.SignalR.Client;
using OeipCommon;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipLiveMedia
{
    public class Room
    {
        //先定为一个MediaServer中多个房间对应一个Nginx,后看有没必要改为一个房间对应一个Nginx
        //public Process NProcess { get; set; } = null;
        public string Name { get; set; } = string.Empty;
        public string Server { get; set; } = string.Empty;
        public int Port { get; set; } = -1;
    }

    public class LiveMedia
    {
        private List<Room> rooms = new List<Room>();
        //先定为一个MediaServer中多个房间对应一个Nginx,后看有没必要改为一个房间对应一个Nginx
        public Process NProcess { get; set; } = null;
        private HubConnection Connection { get; set; }
        private IHubProxy HubProxy { get; set; }
        private string nginxpath = string.Empty;
        private string localServer = "127.0.0.1";

        public void Run()
        {
            var server = ConfigurationManager.AppSettings["server"];
            var port = ConfigurationManager.AppSettings["port"];
            nginxpath = ConfigurationManager.AppSettings["nginxpath"];
            int nport = 1935;//
            int.TryParse(ConfigurationManager.AppSettings["nginxport"], out nport);
            localServer = server;

            string path = Path.Combine(nginxpath, "nginx.exe");
            if (!File.Exists(path))
            {
                LogHelper.LogMessage($"没有找到文件:{path},请修改配置文件.", OeipLogLevel.OEIP_ERROR);
            }
            bool bStart = RunNginx(nport);

            var xuri = $"http://{server}:{(port ?? "80")}";
            Connection = new HubConnection(xuri);
            HubProxy = Connection.CreateHubProxy("OeipMedia");
            HubProxy.On("OnConnect", (string liveServer) =>
            {
                LogHelper.LogMessage($"连接服务器 {liveServer} 成功");
            });
            HubProxy.On("AddRoom", (string roomName) =>
            {
                if (bStart)
                {
                    //发送服务器初始化成功消息
                    HubProxy.Invoke("OnServerAddRoom", roomName, localServer, nport);
                    Room room = new Room();
                    room.Name = roomName;
                    //room.NProcess = process;
                    room.Server = localServer;
                    room.Port = nport;
                    rooms.Add(room);
                    LogHelper.LogMessage($"添加房间 {roomName} 成功");
                }
            });
            HubProxy.On("RemoveRoom", (string roomName) =>
            {
                var room = rooms.Find((p) => p.Name == roomName);
                rooms.Remove(room);
                LogHelper.LogMessage($"删除房间 {roomName} 成功");
            });
            Connection.Start();
        }

        private Process GetProcess(string gamePath)
        {
            var processArray = Process.GetProcesses();
            foreach (var porcess in processArray)
            {
                if (porcess.ProcessName == gamePath)
                {
                    LogHelper.LogMessage("查找到进程:" + gamePath);
                    return porcess;
                }
            }
            return null;
        }

        public bool RunNginx(int port)
        {
            NProcess = GetProcess("nginx");
            if (NProcess != null)
                return true;
            //var gameArguments = " -port=" + port;
            LogHelper.LogMessage($"正在打开端口为{port}的Nginx服务器.");
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = "powershell.exe";//powershell/cmd
            info.UseShellExecute = false;
            info.RedirectStandardInput = true;
            info.RedirectStandardOutput = true;
            info.CreateNoWindow = false;
            NProcess = new Process
            {
                StartInfo = info,
                EnableRaisingEvents = true
            };
            NProcess.Exited += (senderx, args) =>
            {
                HubProxy.Invoke("NginxClose", port);
                LogHelper.LogMessage($"关闭端口{port}为Nginx服务器.");
                try
                {
                    NProcess.Dispose();
                    NProcess = null;
                }
                catch (Exception ex)
                {
                    LogHelper.LogMessageEx("关闭Nginx服务器出错", ex);
                }
            };
            NProcess.Start();
            string cmdStr = "cd " + nginxpath;
            //cmdStr = cmdStr.Replace("\\", "\\\\");
            NProcess.StandardInput.WriteLine(cmdStr);
            NProcess.StandardInput.WriteLine("start nginx.exe");
            LogHelper.LogMessage($"端口为{port}的Nginx服务器打开.");
            return true;
        }

        public void Close()
        {
            if (NProcess != null && !NProcess.HasExited)
            {
                NProcess.Kill();
            }
        }
    }
}
