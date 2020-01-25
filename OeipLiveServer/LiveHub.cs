using Microsoft.AspNet.SignalR;
using Microsoft.AspNet.SignalR.Hubs;
using OeipCommon;
using System.Collections.Generic;
using System.Threading.Tasks;


namespace OeipLiveServer
{
    [HubName("OeipLive")]
    public class LiveHub : Hub
    {
        //客户端每个请求都会生成一个Hub
        public LiveHub()
        {
        }

        public static IHubContext Hub
        {
            get
            {
                return GlobalHost.ConnectionManager.GetHubContext<LiveHub>();
            }
        }
        /// <summary>        
        /// 其直接在OnConnected/OnDisconnected的添加与删除事件还是会导致事件累加
        /// 声明成静态方法可以解决
        /// </summary>
        /// <param name="stream"></param>
        /// <param name="bAdd"></param>
        private static void Instance_OnStreamChangeEvent(StreamDesc stream, bool bAdd)
        {
            //通知此房间所有用户都可以拉流了
            Hub.Clients.Group(stream.User.InRoom.Name)?.OnStreamUpdate(stream.User.Id, stream.Index, bAdd);
        }

        private static void Instance_OnServerCreateEvent(Room room)
        {
            if (room == null)
                return;
            foreach (var user in room.Users)
            {
                SetUserLogin(user);
            }
        }

        private static void SetUserLogin(User user)
        {
            if (user.IsSend)
                return;
            LogHelper.LogMessage($"用户 {user.ConnectId} 分配直播服务器 {user.InRoom.Server}:{user.InRoom.Port}");
            //返回当前用户的结果
            Hub.Clients.Client(user.ConnectId)?.OnLoginRoom(user.Id, user.InRoom.Server, user.InRoom.Port);
            //查找房间里已经存在的流通知用户
            List<StreamDesc> streams = RoomManager.Instance.GetStreams(user);
            foreach (var stream in streams)
            {
                Hub.Clients.Client(user.ConnectId)?.OnStreamUpdate(stream.User.Id, stream.Index, true);
            }
            user.IsSend = true;
        }

        public int LoginRoom(string name, int userId)
        {
            LogHelper.LogMessage($"用户 {Context.ConnectionId} 登陆房间 {name}");
            //查找是否已经开启了
            var room = RoomManager.Instance.GetRoom(name);
            //如果为空，则需要让媒体服务器生成一个直播房间
            if (room == null)
            {
                room = RoomManager.Instance.AddRoom(name);
            }
            if (room == null)
            {
                Clients.Caller?.OnLoginRoom(-1, string.Empty, -1);
                return -1;
            }
            else
            {
                Groups.Add(Context.ConnectionId, name);
                //给用户分配房间以及用户ID
                int uid = RoomManager.Instance.BindUser(name, Context.ConnectionId, userId);
                //通知当前房间所有用户添加新用户                
                Clients.Group(room.Name)?.OnUserChange(Context.ConnectionId, true);
                //如果已经创建直播服务器
                if (room.IsCreate)
                {
                    //返回给用户分配的直播服务器信息
                    User user = RoomManager.Instance.GetUser(Context.ConnectionId);
                    RoomManager.Instance.LockAction(() => { SetUserLogin(user); });
                }
                return uid;
            }
        }

        public int PushStream(int index, bool bVideo, bool bAudio)
        {
            bool result = RoomManager.Instance.AddStream(Context.ConnectionId, index, bVideo, bAudio);
            return result ? 0 : -1;
        }

        public int StopPushStream(int index)
        {
            bool result = RoomManager.Instance.RemoveStream(Context.ConnectionId, index);
            return result ? 0 : -1;
        }

        public string PullStream(int userId, int index)
        {
            //查找是否已经开启了
            var user = RoomManager.Instance.GetUser(userId);
            if (user != null && user.InRoom != null)
            {
                string roomuser = $"{user.InRoom.Name}_{user.Id}_{index}";
                string pullUri = $"{ user.InRoom.Server }:{ user.InRoom.Port}/live/{roomuser}";
                LogHelper.LogMessage($"用户ID {userId} 开始拉流 {pullUri}");
                return roomuser;
            }
            return string.Empty;
        }

        public int StopPullStream(int userId, int index)
        {
            var user = RoomManager.Instance.GetUser(userId);
            if (user != null && user.InRoom != null)
            {
                string roomuser = $"{user.InRoom.Name}_{user.Id}_{index}";
                string pullUri = $"{ user.InRoom.Server }:{ user.InRoom.Port}/live/{roomuser}";
                LogHelper.LogMessage($"用户ID {userId} 停止拉流 {pullUri}");
            }
            return 0;
        }

        public int LogoutRoom()
        {
            var user = RoomManager.Instance.GetUser(Context.ConnectionId);
            Groups.Remove(Context.ConnectionId, user.InRoom.Name);
            //取消绑定用户
            RoomManager.Instance.UnBindUser(Context.ConnectionId);
            //发给用户回调
            Clients.Client(Context.ConnectionId)?.OnLogoutRoom();
            return 0;
        }

        public override Task OnConnected()
        {
            RoomManager.Instance.OnServerCreateEvent += Instance_OnServerCreateEvent;
            RoomManager.Instance.OnStreamChangeEvent += Instance_OnStreamChangeEvent;
            LogHelper.LogMessage($"用户 {Context.ConnectionId} 连接成功");
            RoomManager.Instance.AddUser(Context.ConnectionId);
            return base.OnConnected();
        }

        public override Task OnDisconnected(bool stopCalled)
        {
            RoomManager.Instance.OnServerCreateEvent -= Instance_OnServerCreateEvent;
            RoomManager.Instance.OnStreamChangeEvent -= Instance_OnStreamChangeEvent;
            RoomManager.Instance.DeleteUser(Context.ConnectionId);
            LogHelper.LogMessage($"用户 {Context.ConnectionId} 失去连接");
            return base.OnDisconnected(stopCalled);
        }
    }
}
