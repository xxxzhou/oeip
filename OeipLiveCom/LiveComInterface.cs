using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OeipLiveCom
{
    [Guid("6451E50D-DB0B-496D-B0AB-7902BA48D669")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    [ComVisible(true)]
    public interface IOeipLiveCallBack
    {
        [DispId(1)]
        void OnInitRoom(int code);
        [DispId(2)]
        void OnUserChange(int userId, bool bAdd);
        [DispId(3)]
        void OnLoginRoom(int code, string server, int port);
        [DispId(4)]
        void OnStreamUpdate(int userId, int index, bool bAdd);
        [DispId(5)]
        void OnLogoutRoom();
        [DispId(6)]
        void OnOperateResult(int operate, int code, string message);
        [DispId(7)]
        void Dispose();
    }

    [Guid("F16C2C38-FF79-463F-BD8A-D9F2C799348B")]
    [ComVisible(true)]
    public interface IOeipLiveClient
    {
        //IOeipLiveCallBack liveBack { get; set; }
        [DispId(1)]
        void SetLiveCallBack(ref IOeipLiveCallBack liveBack);
        //int Add(int a, int b);
        [DispId(2)]
        bool InitRoom(string uri);
        [DispId(3)]
        int LoginRoom(string roomName, int userId);
        [DispId(4)]
        int PushStream(int index, bool bVideo, bool bAudio);
        [DispId(5)]
        int StopPushStream(int index);
        [DispId(6)]
        string PullStream(int userId, int index);
        [DispId(7)]
        int StopPullStream(int userId, int index);
        [DispId(8)]
        int LogoutRoom();
        [DispId(9)]
        void Shutdown();
    }
}
