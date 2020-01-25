
using OeipCommon;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace OeipLiveServer
{
    /// <summary>
    /// 媒体服务器
    /// </summary>
    public class MediaServer
    {
        public string ConnectId { get; set; } = string.Empty;
    }

    public class User
    {
        public string ConnectId { get; set; } = string.Empty;
        public Room InRoom { get; set; } = null;
        public int Id { get; set; } = 0;
        /// <summary>
        /// 是否已经发送OnLoginRoom回调,有二种情况,第一个登陆房间的人,
        /// 需要等房间创建然后发送OnLoginRoom,后面房间的人则可以直接发送OnLoginRoom
        /// 但是这个回调只发送一次，这个标记是否已经发送
        /// </summary>
        public bool IsSend = false;
    }

    public class StreamDesc
    {
        public User User { get; set; } = null;
        public int Index { get; set; } = 0;
        public bool bVideo { get; set; } = false;
        public bool bAudio { get; set; } = false;
    }

    public class Room
    {
        public string Name { get; set; } = string.Empty;
        //在这限定只能RoomManager访问，此资源有多个线程访问
        public List<User> Users { get; set; } = new List<User>();
        public MediaServer Media { get; set; } = null;
        public string Server { get; set; } = string.Empty;
        public int Port { get; set; } = -1;
        //是否已经创建直播服务器
        public bool IsCreate
        {
            get
            {
                return !string.IsNullOrEmpty(Server) && Port > 0;
            }
        }
    }

    //为了避免多线程操作资源出问题,规定如下二点
    //1 事件调用尽量不包含在lock中,特别是发出去给外部人员调用，自己控制不了
    //2 如果事件在lock里，就要控制事件引用方法在本程序集运行，不要外部程序集可见，并且在本程序集调用避免lock in lock
    //这个类里操作大部分要上锁
    public class RoomManager
    {
        private object obj = new object();
        //所有事件发出去后尽量不要在里面回调RoomManager.Instance里的方法，如果回调，注意lock，事件调用也可以尽量不包含在lock中
        internal event Action<Room, User, bool> OnRoomChangeEvent;
        //public event Func<MediaServer, string, bool, Task> OnMediaServerEvent;
        //当需要删除或是添加服务器的时候
        internal event Action<MediaServer, string, bool> OnMediaServerEvent;
        //媒体服务器返回对应sever/port后
        internal event Action<Room> OnServerCreateEvent;
        //用户推流
        internal event Action<StreamDesc, bool> OnStreamChangeEvent;
        protected RoomManager() { }
        protected static RoomManager instance = null;
        public static RoomManager Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new RoomManager();
                }
                return instance;
            }
        }

        private List<MediaServer> mediaServers = new List<MediaServer>();
        private List<User> users = new List<User>();
        private List<Room> rooms = new List<Room>();
        private List<StreamDesc> streams = new List<StreamDesc>();

        public string SelfHost
        {
            get
            {
                var server = ConfigurationManager.AppSettings["server"];
                var port = ConfigurationManager.AppSettings["port"];
                string url = string.Format("http://{0}:{1}", server, port);
                return url;
            }
        }

        /// <summary>
        /// 添加一个房间
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Room AddRoom(string name)
        {
            lock (obj)
            {
                //查找媒体服务器里最适用的那一个
                if (mediaServers.Count <= 0)
                {
                    LogHelper.LogMessage($"还没有任何媒体服务器添加进来,不能分配房间 {name}", OeipLogLevel.OEIP_ERROR);
                    return null;
                }
                Room room = new Room();
                room.Name = name;
                room.Server = string.Empty;
                room.Port = -1;
                rooms.Add(room);
                //在这逻辑先固定选第一台，后期根据实际情况来看
                MediaServer mediaServer = mediaServers[0];
                room.Media = mediaServer;
                //告诉媒体服务器端，需要添加一个房间供直播
                OnMediaChange(mediaServer, name, true);
                return room;
            }
        }
        /// <summary>
        /// 添加用户
        /// </summary>
        /// <param name="connectId"></param>
        public void AddUser(string connectId)
        {
            lock (obj)
            {
                User user = new User();
                user.ConnectId = connectId;
                user.IsSend = false;
                users.Add(user);
            }
        }
        public void DeleteUser(string connectId)
        {
            lock (obj)
            {
                User user = getUser(connectId);
                if (user != null)
                {
                    //如果用户还有房间信息，先取消房间与用户的关系
                    if (user.InRoom != null)
                    {
                        unBindUser(user);
                    }
                    users.Remove(user);
                }
            }
        }
        /// <summary>
        /// 添加媒体服务器
        /// </summary>
        /// <param name="connectId"></param>
        public void AddMediaServe(string connectId)
        {
            lock (obj)
            {
                MediaServer mediaServer = new MediaServer();
                mediaServer.ConnectId = connectId;
                mediaServers.Add(mediaServer);
            }
        }
        public void RemoveMediaServer(string connectId)
        {
            lock (obj)
            {
                var mediaServer = mediaServers.Find((p) => p.ConnectId == connectId);
                if (mediaServer != null)
                {
                    foreach (var room in rooms)
                    {
                        if (room.Media == mediaServer)
                            room.Media = null;

                    }
                    mediaServers.Remove(mediaServer);
                }
            }
        }
        /// <summary>
        /// 用户推流记录
        /// </summary>
        /// <param name="connectId"></param>
        /// <param name="index"></param>
        /// <param name="bVideo"></param>
        /// <param name="bAudio"></param>
        /// <returns></returns>
        public bool AddStream(string connectId, int index, bool bVideo, bool bAudio)
        {
            lock (obj)
            {
                User user = getUser(connectId);
                if (user == null)
                {
                    LogHelper.LogMessage($"用户{connectId}没有注册，不能添加流.", OeipLogLevel.OEIP_WARN);
                    return false;
                }
                bool bHave = streams.Exists((p) => p.User == user && p.Index == index);
                if (bHave)
                {
                    LogHelper.LogMessage($"用户{connectId}已经添加流{index}，不能再次添加.", OeipLogLevel.OEIP_WARN);
                    return false;
                }
                if (user != null)
                {
                    StreamDesc stream = new StreamDesc();
                    stream.bAudio = bAudio;
                    stream.bVideo = bVideo;
                    stream.Index = index;
                    stream.User = user;
                    streams.Add(stream);
                    OnStreamChange(stream, true);
                    return true;
                }
            }
            return false;
        }
        public bool RemoveStream(string connectId, int index)
        {
            lock (obj)
            {
                User user = getUser(connectId);
                var stream = streams.Find(p => p.Index == index && p.User == user);
                if (stream == null)
                {
                    LogHelper.LogMessage($"用户{connectId}没有找到流，不能移除流.", OeipLogLevel.OEIP_WARN);
                    return false;
                }
                if (stream != null)
                {
                    OnStreamChange(stream, false);
                    streams.Remove(stream);
                    return true;
                }
            }
            return false;
        }

        private void OnRoomChange(Room room, User user, bool bAdd)
        {
            var amessage = bAdd ? "添加" : "删除";
            LogHelper.LogMessage($"房间 {room.Name} {amessage}用户 {user.ConnectId}");
            OnRoomChangeEvent?.Invoke(room, user, bAdd);
        }

        private void OnMediaChange(MediaServer media, string roomName, bool bAdd)
        {
            var amessage = bAdd ? "添加" : "删除";
            LogHelper.LogMessage($"媒体服务器 {media.ConnectId} {amessage}房间 {roomName}");
            OnMediaServerEvent?.Invoke(media, roomName, bAdd);
        }

        private void OnServerCreate(Room room)
        {
            OnServerCreateEvent?.Invoke(room);
        }

        private void OnStreamChange(StreamDesc desc, bool bAdd)
        {
            var amessage = bAdd ? "添加" : "删除";
            LogHelper.LogMessage($"房间 {desc.User.InRoom.Name} {amessage}流,流用户:{desc.User.ConnectId}-流索引:{desc.Index}");
            if (bAdd)
            {
                string pullUri = $"{ desc.User.InRoom.Server }:{ desc.User.InRoom.Port}/live/{desc.User.InRoom.Name}_{desc.User.Id}_{desc.Index}";
                LogHelper.LogMessage($"生成地址:{ pullUri}");
            }
            OnStreamChangeEvent?.Invoke(desc, bAdd);
        }

        /// <summary>
        /// 绑定房间与用户
        /// </summary>
        /// <param name="roomName">房间名</param>
        /// <param name="userCid">用户连接ID</param>
        /// <param name="userId">用户传入ID,如果已有或是小于0,生成新的ID</param>
        /// <returns>返回可能新生成的ID</returns>
        public int BindUser(string roomName, string userCid, int userId)
        {
            lock (obj)
            {
                var roomx = getRoom(roomName);
                var userx = getUser(userCid);
                if (roomx == null)
                {
                    LogHelper.LogMessage($"房间 {roomName} 没找到", OeipLogLevel.OEIP_WARN);
                    return -1;
                }
                if (userx == null)
                {
                    LogHelper.LogMessage($"用户 {userCid} 没找到", OeipLogLevel.OEIP_WARN);
                    return -1;
                }
                if (userx.InRoom != null)
                {
                    LogHelper.LogMessage($"用户 {userCid} 已经在 {roomName}", OeipLogLevel.OEIP_WARN);
                    return -1;
                }
                int autoUserId = userId;
                bool bHaveId = roomx.Users.Exists((p) => p.Id == userId);
                if (bHaveId || autoUserId <= 0)
                {
                    int maxId = roomx.Users.Select((p) => p.Id).Max();
                    if (maxId == 0)
                        maxId = 1024;
                    autoUserId = maxId + 1;
                }
                userx.Id = autoUserId;
                roomx.Users.Add(userx);
                userx.InRoom = roomx;
                LogHelper.LogMessage($"房间 {roomx.Name} 给用户 {userCid} 分配ID {userx.Id}");
                OnRoomChange(roomx, userx, true);
                return userx.Id;
            }
        }

        /// <summary>
        /// 取消房间里对应用户
        /// </summary>
        /// <param name="user"></param>
        /// <returns></returns>
        private int unBindUser(User user)
        {
            if (user == null)
            {
                LogHelper.LogMessage($"用户 {user.ConnectId} 没找到", OeipLogLevel.OEIP_WARN);
                return -1;
            }
            //正常情况下，在用户退出前，有此用户的streams应该都删除了,此处为删除非常情况下还保留的
            foreach (var stream in streams)
            {
                if (stream.User == user)
                {
                    OnStreamChange(stream, false);
                }
            }
            streams.RemoveAll(p => p.User == user);
            //关闭对应用户房间
            if (user.InRoom == null)
            {
                LogHelper.LogMessage($"用户 {user.ConnectId} 没在房间里,不能取消房间绑定", OeipLogLevel.OEIP_WARN);
                return -1;
            }
            //房间去掉用户
            user.InRoom.Users.Remove(user);
            int count = user.InRoom.Users.Count;
            OnRoomChange(user.InRoom, user, false);
            //如果房间里没有用户了,开始清除相应房间操作
            if (count == 0)
            {
                OnMediaChange(user.InRoom.Media, user.InRoom.Name, false);
                rooms.Remove(user.InRoom);
            }
            user.InRoom = null;
            user.IsSend = false;
            return 0;
        }

        public int UnBindUser(string connectId)
        {
            lock (obj)
            {
                var user = getUser(connectId);
                return unBindUser(user);
            }
        }

        public void LockAction(Action action)
        {
            lock (obj)
            {
                action();
            }
        }
        /// <summary>
        /// 给房间分配媒体服务器的服务器地址与端口
        /// </summary>
        /// <param name="roomName"></param>
        /// <param name="server"></param>
        /// <param name="port"></param>
        public void BindRoom(string roomName, string server, int port)
        {
            lock (obj)
            {
                Room room = getRoom(roomName);
                if (room != null)
                {
                    room.Server = server;
                    room.Port = port;
                }
                //给要推流的安排上
                OnServerCreate(room);
            }
        }

        private Room getRoom(string name)
        {
            var room = rooms.Find((p) => p.Name == name);
            return room;
        }

        private User getUser(string connectId)
        {
            var user = users.Find((p) => p.ConnectId == connectId);
            return user;
        }

        public Room GetRoom(string name)
        {
            lock (obj)
            {
                return getRoom(name);
            }
        }

        public User GetUser(string connectId)
        {
            lock (obj)
            {
                return getUser(connectId);
            }
        }

        public User GetUser(int userId)
        {
            lock (obj)
            {
                var user = users.Find(p => p.Id == userId);
                return user;
            }
        }

        /// <summary>
        /// 确认这个类调用同步之下
        /// </summary>
        /// <param name="user"></param>
        /// <returns></returns>
        public List<StreamDesc> GetStreams(User user)
        {
            List<StreamDesc> ustreams = new List<StreamDesc>();
            foreach (var stream in streams)
            {
                if (user.InRoom.Users.Exists(p => { return p == stream.User; }))
                {
                    ustreams.Add(stream);
                }
            }
            return ustreams;
        }
    }
}
