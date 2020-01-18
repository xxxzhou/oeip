using Microsoft.AspNet.SignalR.Client;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

//计算机\HKEY_CLASSES_ROOT\WOW6432Node\CLSID\{3262E386-7B1D-4EF9-8818-8506357EA086}\Control
//C:\Windows\Microsoft.NET\Framework\v4.0.30319\RegAsm.exe ..Path\ZmfLiveCom.dll

namespace OeipLiveCom
{

    //[ComSourceInterfaces(typeof(IOeipLiveClient))]
    [ClassInterface(ClassInterfaceType.None)]
    [ProgId("OeipLiveCom.OeipLiveClient")]
    [Guid("3262E386-7B1D-4EF9-8818-8506357EA086")]
    [ComVisible(true)]//确定相应的liveBack属性对于C++程序可见
    public class OeipLiveClient : IOeipLiveClient
    {
        private HubConnection Connection { get; set; }
        private IHubProxy HubProxy { get; set; }

        private string roomName = string.Empty;
        #region IOeipLiveCallBack
        public IOeipLiveCallBack liveBack { get; set; }

        public int Add(int a, int b)
        {
            return a + b;
        }

        public bool InitRoom(string uri)
        {
            bool bInit = false;
            try
            {
                if (liveBack == null)
                {
                    return false;
                }
                Connection = new HubConnection(uri);
                HubProxy = Connection.CreateHubProxy("OeipLive");
                HubProxy.On(("OnConnect"), () =>
                {
                    //发送服务器初始化成功消息
                    HubProxy.Invoke("UserInit");
                });
                HubProxy.On("OnLoginRoom", (int code, string server, int port) =>
                {
                    liveBack.OnLoginRoom(code, server, port);
                });
                HubProxy.On("OnUserChange", (int userId, bool bAdd) =>
                {
                    liveBack.OnUserChange(userId, bAdd);
                });
                HubProxy.On("OnStreamUpdate", (int userId, int index, bool bAdd) =>
                {
                    liveBack.OnStreamUpdate(userId, index, bAdd);
                });
                HubProxy.On("OnLogoutRoom", () =>
                {
                    if (liveBack != null)
                        liveBack.OnLogoutRoom();
                });
                bInit = ConnectServer().Result;
            }
            catch (Exception)
            {
                bInit = false;
            }
            return bInit;
        }

        public void Invoke(string serverMethod, params object[] objs)
        {
            Task.Run(() =>
            {
                try
                {
                    HubProxy.Invoke(serverMethod, objs);
                }
                catch (Exception ex)
                {
                    if (liveBack != null)
                    {
                        liveBack.OnOperateResult(11, -1, ex.Message);
                    }
                }
            });
        }

        public T Invoke<T>(string serverMethod, params object[] objs)
        {
            try
            {
                //Task<T> task = HubProxy.Invoke<T>(serverMethod, objs);
                //T result = task.Result;
                T result = HubProxy.Invoke<T>(serverMethod, objs).Result;
                return result;
            }
            catch (Exception ex)
            {
                if (liveBack != null)
                {
                    liveBack.OnOperateResult(11, -1, ex.Message);
                }
            }
            return default(T);
        }

        public int LoginRoom(string roomName, int userId)
        {
            this.roomName = roomName;
            //发送服务器消息
            int result = Invoke<int>("LoginRoom", roomName, userId);
            return userId;
        }

        private Task<bool> ConnectServer()
        {
            return Task.Run<bool>(() =>
            {
                Connection.Start();
                return true;
            });
        }

        public int PushStream(int index, bool bVideo, bool bAudio)
        {
            //int result = Invoke<int>("PushStream", index, bVideo, bAudio);
            //return result;
            Invoke("PushStream", index, bVideo, bAudio);
            return 0;
        }

        public int StopPushStream(int index)
        {
            //int result = Invoke<int>("StopPushStream", index);
            //return result;
            Invoke("StopPushStream", index);
            return 0;
        }

        public string PullStream(int userId, int index)
        {
            //var result = Invoke<string>("PullStream", userId, index);
            Invoke("PullStream", userId, index);
            var result = $"{roomName}_{userId}_{index}";
            return result;
        }

        public int StopPullStream(int userId, int index)
        {
            Invoke("StopPullStream", userId, index);
            return 0;
        }

        public int LogoutRoom()
        {
            Invoke("LogoutRoom");
            return 0;
        }
        public void Shutdown()
        {
            try
            {
                Connection.Stop(new TimeSpan(1000));
                Connection.Dispose();
            }
            catch (Exception ex)
            {
                //liveBack.OnOperateResult(12, -1, ex.Message);
            }
        }
        #endregion
        //VS请用管理员模式打开
        [ComRegisterFunction]
        static void ComRegister(Type t)
        {
            ComRegisterHelper.Register(t);
        }

        [ComUnregisterFunction]
        static void ComUnregister(Type t)
        {
            ComRegisterHelper.Unregister(t);
        }


    }
}
