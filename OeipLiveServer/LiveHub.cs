using Microsoft.AspNet.SignalR;
using Microsoft.AspNet.SignalR.Hubs;
using OeipCommon;
using System.Threading.Tasks;


namespace OeipLiveServer
{
    [HubName("OeipLive")]
    public class LiveHub : Hub
    {
        public LiveHub()
        {
        }

        private void Instance_OnStreamChangeEvent(StreamDesc stream, bool bAdd)
        {
            //通知此房间所有用户都可以拉流了
            Clients.Group(stream.User.InRoom.Name)?.OnStreamUpdate(stream.User.Id, stream.Index, bAdd);
        }

        private void Instance_OnServerCreateEvent(Room room)
        {
            if (room == null)
                return;
            foreach (var user in room.Users)
            {
                if (!user.IsSend)
                {
                    LogHelper.LogMessage($"用户 {Context.ConnectionId} 分配直播服务器 {room.Server}:{room.Port}");
                    //返回当前用户的结果
                    Clients.Client(user.ConnectId)?.OnLoginRoom(0, room.Server, room.Port);
                    user.IsSend = true;
                }
            }
        }

        public bool UserInit()
        {
            LogHelper.LogMessage($"用户 {Context.ConnectionId} 初始化成功");
            RoomManager.Instance.AddUser(Context.ConnectionId);
            return true;
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
                    User user = RoomManager.Instance.GetUser(Context.ConnectionId);
                    if (user != null && !user.IsSend)
                    {
                        LogHelper.LogMessage($"用户 {Context.ConnectionId} 分配直播服务器 {room.Server}:{room.Port}");
                        Clients.Client(Context.ConnectionId)?.OnLoginRoom(0, room.Server, room.Port);
                        user.IsSend = true;
                    }
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
            //注意,Clients.Caller的调用要在base.OnConnected()后面
            return Task.Run(() =>
            {
                base.OnConnected();
                Clients.Caller.OnConnect();
            });
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
