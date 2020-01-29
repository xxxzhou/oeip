using Microsoft.AspNet.SignalR;
using Microsoft.AspNet.SignalR.Hubs;
using OeipCommon;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;


namespace OeipLiveServer
{
    [HubName("OeipMedia")]
    public class MediaHub : Hub
    {
        public MediaHub()
        {
        }

        public static IHubContext Hub
        {
            get
            {
                return GlobalHost.ConnectionManager.GetHubContext<MediaHub>();
            }
        }

        //通知直播服务器，添加相应媒体服务器数据
        public void OnServerAddRoom(string roomName, string server, int port)
        {
            LogHelper.LogMessage($"媒体服务器 分配房间 {roomName} {server}:{port} 添加成功");
            RoomManager.Instance.BindRoom(roomName, server, port);
        }

        public void NginxClose(int port)
        {
        }

        /// <summary>
        /// 发送媒体服务器,添加或是删除一个组(就是一个房间)
        /// </summary>
        /// <param name="media"></param>
        /// <param name="roomName"></param>
        /// <param name="bAdd"></param>
        private void Instance_OnMediaServerEvent(MediaServer media, string roomName, bool bAdd)
        {
            if (bAdd)
            {
                Clients.Client(media.ConnectId).AddRoom(roomName);
            }
            else
            {
                Clients.Client(media.ConnectId).RemoveRoom(roomName);
            }
        }

        public override Task OnConnected()
        {
            //RoomManager.Instance.OnMediaServerEvent += Instance_OnMediaServerEvent;
            LogHelper.LogMessage($"媒体服务器 {Context.ConnectionId} 连接成功");
            RoomManager.Instance.AddMediaServe(Context.ConnectionId);
            Clients.Caller.OnConnect(RoomManager.Instance.SelfHost);
            return base.OnConnected();
        }

        public override Task OnDisconnected(bool stopCalled)
        {
           // RoomManager.Instance.OnMediaServerEvent -= Instance_OnMediaServerEvent;
            LogHelper.LogMessage($"媒体服务器 {Context.ConnectionId} 丢失连接");
            RoomManager.Instance.RemoveMediaServer(Context.ConnectionId);
            return base.OnDisconnected(stopCalled);
        }
    }
}
