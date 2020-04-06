using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    public class OeipMediaPlay
    {
        protected OnOperateDelegate onOperateDelegate;
        public event Action<bool, bool> OnOpenEvent;
        //protected event OnOperateDelegate OnOperateEvent;
        protected OnVideoFrameDelegate onVideoFrameDelegate;
        public event OnVideoFrameDelegate OnVideoFrameEvent;
        protected OnAudioFrameDelegate onAudioFrameDelegate;
        public event OnAudioFrameDelegate OnAudioFrameEvent;

        public int MediaId { get; private set; } = -1;
        public bool IsOpen { get; private set; } = false;
        public bool IsVideo { get; private set; } = false;
        public bool IsAudiio { get; private set; } = false;
        private OeipVideoEncoder videoInfo = new OeipVideoEncoder();
        private OeipAudioEncoder audioInfo = new OeipAudioEncoder();
        public ref OeipVideoEncoder VideoInfo
        {
            get
            {
                return ref videoInfo;
            }
        }

        public ref OeipAudioEncoder AudioInfo
        {
            get
            {
                return ref audioInfo;
            }
        }

        public OeipMediaPlay()
        {
            onOperateDelegate = new OnOperateDelegate(OnOperateHandle);
            onVideoFrameDelegate = new OnVideoFrameDelegate(OnVideoFrameHandle);
            onAudioFrameDelegate = new OnAudioFrameDelegate(OnAudioFrameHandle);
        }

        public void SetMediaId(int id)
        {
            this.MediaId = id;
            OeipHelper.setReadOperateAction(this.MediaId, onOperateDelegate);
            OeipHelper.setVideoDataAction(this.MediaId, onVideoFrameDelegate);
            OeipHelper.setAudioDataAction(this.MediaId, onAudioFrameDelegate);
        }

        public void Open(string uri, bool bPlayAudio)
        {
            IsOpen = OeipHelper.openReadMedia(this.MediaId, uri, bPlayAudio) >= 0;
        }

        public void Close()
        {
            IsOpen = false;
            OeipHelper.closeReadMedia(this.MediaId);
        }

        private void OnOperateHandle(int type, int code)
        {
            if (code < 0)
            {
                return;
            }
            if (type == (int)OeipFFmpegMode.OEIP_DECODER_OPEN)
            {
                IsAudiio = OeipHelper.getMediaAudioInfo(MediaId, ref audioInfo);
                IsVideo = OeipHelper.getMediaVideoInfo(MediaId, ref videoInfo);
                OnOpenEvent?.Invoke(IsVideo, IsVideo);
            }
        }

        private void OnVideoFrameHandle(OeipVideoFrame videoFrame)
        {
            OnVideoFrameEvent?.Invoke(videoFrame);
        }

        private void OnAudioFrameHandle(OeipAudioFrame audioFrame)
        {
            OnAudioFrameEvent?.Invoke(audioFrame);
        }
    }
}
