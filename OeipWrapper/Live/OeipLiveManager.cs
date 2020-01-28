using OeipCommon;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper.Live
{
    public class OeipLiveManager : MSingleton<OeipLiveManager>
    {
        public bool IsInit { get; private set; }
        public int UserId { get; private set; } = 0;
        private LiveBackWrapper liveBackWrapper = new LiveBackWrapper();
        private OeipVideoFrame mainVideoFrame = new OeipVideoFrame();
        private OeipVideoFrame auxVideoFrame = new OeipVideoFrame();

        private object obj = new object();
        private bool bLogin = false;

        #region LiveBackWrapper 转播 
        /// <summary>
        /// 房间初始化回调
        /// </summary>
        public event OnInitRoomDelegate OnInitRoomEvent;
        /// <summary>
        /// 登陆房间回调
        /// </summary>
        public event OnLoginRoomDelegate OnLoginRoomEvent;
        /// <summary>
        /// 当登陆的房间有用户变化回调相应变化
        /// </summary>
        public event OnUserChangeDelegate OnUserChangeEvent;
        /// <summary>
        /// 当登陆的房间用推流变化回调
        /// </summary>
        public event OnStreamUpdateDelegate OnStreamUpdateEvent;
        /// <summary>
        /// 拉流的视频数据回调
        /// </summary>
        public event OnVideoFrameDelegate OnVideoFrameEvent;
        /// <summary>
        /// 拉流的音频数据回调
        /// </summary>
        public event OnAudioFrameDelegate OnAudioFrameEvent;
        /// <summary>
        /// 用户登陆房间回调
        /// </summary>
        public event OnLogoutRoomDelegate OnLogoutRoomEvent;
        /// <summary>
        /// 用户操作数据各种回调，可能是错误警告或各种信息
        /// </summary>
        public event OnOperateResultDelegate OnOperateResultEvent;
        /// <summary>
        /// 推流成功回调
        /// </summary>
        public event OnPushStreamDelegate OnPushStreamEvent;
        /// <summary>
        /// 拉流成功回调
        /// </summary>
        public event OnPullStreamDelegate OnPullStreamEvent;
        #endregion

        protected override void Init()
        {
            liveBackWrapper.onInitRoomDelegate = new OnInitRoomDelegate(OnInitRoom);
            liveBackWrapper.onLoginRoomDelegate = new OnLoginRoomDelegate(OnLoginRoom);
            liveBackWrapper.onUserChangeDelegate = new OnUserChangeDelegate(OnUserChange);
            liveBackWrapper.onStreamUpdateDelegate = new OnStreamUpdateDelegate(OnStreamUpdate);
            liveBackWrapper.onVideoFrameDelegate = new OnVideoFrameDelegate(OnVideoFrame);
            liveBackWrapper.onAudioFrameDelegate = new OnAudioFrameDelegate(OnAudioFrame);
            liveBackWrapper.onLogoutRoomDelegate = new OnLogoutRoomDelegate(OnLogoutRoom);
            liveBackWrapper.onOperateResultDelegate = new OnOperateResultDelegate(OnOperateResult);
            liveBackWrapper.onPushStreamDelegate = new OnPushStreamDelegate(OnPushStream);
            liveBackWrapper.onPullStreamDelegate = new OnPullStreamDelegate(OnPullStream);

            OeipLiveHelper.initOeipLive();
            IsInit = OeipLiveHelper.initLiveRoomWrapper(ref OeipManager.Instance.LiveCtx, ref liveBackWrapper);
            //Span
        }

        public override void Close()
        {
            OeipLiveHelper.logoutRoom();
            OeipLiveHelper.shutdownOeipLive();
        }

        public bool LoginRoom(string roomName, int userId)
        {
            this.UserId = userId;
            bLogin = OeipLiveHelper.loginRoom(roomName, userId);
            return bLogin;
        }

        public bool PushStream(int index, ref OeipPushSetting setting)
        {
            if (!bLogin)
                return false;
            return OeipLiveHelper.pushStream(index, ref setting);
        }

        public bool PushVideoFrame(int index, IntPtr data, int width, int height, OeipYUVFMT fmt)
        {
            lock (obj)
            {
                if (!bLogin)
                    return false;
                ref OeipVideoFrame videoFrame = ref mainVideoFrame;
                if (index == 1)
                    videoFrame = ref auxVideoFrame;
                OeipLiveHelper.getVideoFrame(data, width, height, fmt, ref videoFrame);
                return OeipLiveHelper.pushVideoFrame(index, ref videoFrame);
            }
        }

        public bool PushAudioFrame(int index, ref OeipAudioFrame audioFrame)
        {
            lock (obj)
            {
                if (!bLogin)
                    return false;
                return OeipLiveHelper.pushAudioFrame(index, ref audioFrame);
            }
        }

        public bool StopPushStream(int index)
        {
            return OeipLiveHelper.stopPushStream(index);
        }

        public bool PullStream(int userId, int index)
        {
            return OeipLiveHelper.pullStream(userId, index);
        }

        public bool StopPullStream(int userId, int index)
        {
            return OeipLiveHelper.stopPullStream(userId, index);
        }

        public bool LogoutRoom()
        {
            lock (obj)
            {
                bLogin = false;
                return OeipLiveHelper.logoutRoom();
            }
        }

        #region LiveBackWrapper 转播 
        internal void OnInitRoom(int code)
        {
            OnInitRoomEvent?.Invoke(code);
        }

        internal void OnLoginRoom(int code, int userid)
        {
            UserId = userid;
            OnLoginRoomEvent?.Invoke(code, userid);
        }

        internal void OnUserChange(int userId, bool bAdd)
        {
            OnUserChangeEvent?.Invoke(userId, bAdd);
        }

        internal void OnStreamUpdate(int userId, int index, bool bAdd)
        {
            OnStreamUpdateEvent?.Invoke(userId, index, bAdd);
        }

        internal void OnVideoFrame(int userId, int index, OeipVideoFrame videoFrame)
        {
            OnVideoFrameEvent?.Invoke(userId, index, videoFrame);
        }

        internal void OnAudioFrame(int userId, int index, OeipAudioFrame audioFrame)
        {
            OnAudioFrameEvent?.Invoke(userId, index, audioFrame);
        }

        internal void OnLogoutRoom(int code)
        {
            OnLogoutRoomEvent?.Invoke(code);
        }

        internal void OnOperateResult(int operate, int code, string message)
        {
            OnOperateResultEvent?.Invoke(operate, code, message);
        }

        internal void OnPushStream(int index, int code)
        {
            OnPushStreamEvent?.Invoke(index, code);
        }

        internal void OnPullStream(int userId, int index, int code)
        {
            OnPullStreamEvent?.Invoke(userId, index, code);
        }
        #endregion
    }
}
