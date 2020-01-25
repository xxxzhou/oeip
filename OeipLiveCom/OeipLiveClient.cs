using Microsoft.AspNet.SignalR.Client;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
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
    public class OeipLiveClient : IOeipLiveClient, IDisposable
    {
        private HubConnection Connection { get; set; }
        private IHubProxy HubProxy { get; set; }

        private string roomName = string.Empty;
        private bool disposed = false;
        #region IOeipLiveCallBack
        //IOeipLiveCallBack对应的是C++实现
        private IOeipLiveCallBack liveBack = null;
        private List<IDisposable> onActions = new List<IDisposable>();

        public void SetLiveCallBack(ref IOeipLiveCallBack liveBack)
        {
            //GC.KeepAlive(liveBack);
            this.liveBack = liveBack;
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
                onActions.Add(HubProxy.On("OnLoginRoom", (int code, string server, int port) =>
                {
                    liveBack?.OnLoginRoom(code, server, port);
                }));
                onActions.Add(HubProxy.On("OnUserChange", (int userId, bool bAdd) =>
                {
                    liveBack?.OnUserChange(userId, bAdd);
                }));
                onActions.Add(HubProxy.On("OnStreamUpdate", (int userId, int index, bool bAdd) =>
                {
                    liveBack?.OnStreamUpdate(userId, index, bAdd);
                }));
                onActions.Add(HubProxy.On("OnLogoutRoom", () =>
                {
                    liveBack?.OnLogoutRoom();
                }));
                Connection.Start().Wait();
                bInit = true;
            }
            catch (Exception)
            {
                bInit = false;
            }
            return bInit;
        }

        //调用本方法的线程与执行异步
        public async Task InvokeAsync(string serverMethod, params object[] objs)
        {
            try
            {
                if (Connection != null && Connection.State == ConnectionState.Connected)
                    await HubProxy.Invoke(serverMethod, objs);
            }
            catch (Exception ex)
            {
                liveBack?.OnOperateResult(11, -1, ex.Message);
            }
        }

        public void Invoke(string serverMethod, params object[] objs)
        {
            InvokeAsync(serverMethod, objs).Wait(3000);
        }

        //调用本方法的线程与执行同步
        public void InvokeSync(string serverMethod, params object[] objs)
        {
            try
            {
                if (Connection != null && Connection.State == ConnectionState.Connected)
                    HubProxy.Invoke(serverMethod, objs).Wait(3000);
            }
            catch (Exception ex)
            {
                liveBack?.OnOperateResult(11, -1, ex.Message);
            }
        }

        public T Invoke<T>(string serverMethod, params object[] objs)
        {
            try
            {
                if (Connection != null && Connection.State == ConnectionState.Connected)
                {
                    T result = HubProxy.Invoke<T>(serverMethod, objs).Result;
                    return result;
                }
            }
            catch (Exception ex)
            {
                liveBack?.OnOperateResult(11, -1, ex.Message);
            }
            return default(T);
        }

        public int LoginRoom(string roomName, int userId)
        {
            this.roomName = roomName;
            //发送服务器消息 暂时来看,直接Task<int>.Result会卡住，原因不明
            //int result = Invoke<int>("LoginRoom", roomName, userId);
            Invoke("LoginRoom", roomName, userId);
            return userId;
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
            InvokeSync("LogoutRoom");
            return 0;
        }
        public virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;     
            liveBack?.Dispose();
            liveBack = null;
            if (disposing)
            {
                foreach (var action in onActions)
                {
                    action.Dispose();
                }
                onActions.Clear();
                Connection.Stop(new TimeSpan(1000));
                Connection.Dispose();
            }
            disposed = true;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
            GC.Collect();
            //挂起当前线程，直到处理终结器队列的线程清空该队列为止。
            GC.WaitForPendingFinalizers();
        }

        public void Shutdown()
        {
            try
            {
                Dispose();
            }
            catch (Exception)
            {
                //liveBack?.OnOperateResult(12, -1, ex.Message);
            }
        }

        ~OeipLiveClient()
        {
            Dispose(false);
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
