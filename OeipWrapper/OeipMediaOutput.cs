using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OeipWrapper
{
    public class OeipMediaOutput
    {
        public int MediaId { get; private set; } = -1;

        protected OnOperateDelegate onOperateDelegate;
        //protected event Action<bool, bool> OnOpenEvent;
        public bool IsOpen { get; private set; } = false;

        public OeipMediaOutput()
        {
            onOperateDelegate = new OnOperateDelegate(OnOperateHandle);
        }

        public void SetMediaId(int id)
        {
            this.MediaId = id;
            OeipHelper.setWriteOperateAction(this.MediaId, onOperateDelegate);
        }

        public void SetVideoEncoder(OeipVideoEncoder videoInfo)
        {
            OeipHelper.setVideoEncoder(this.MediaId, videoInfo);
        }

        public void SetAudioEncoder(OeipAudioEncoder audioInfo)
        {
            OeipHelper.setAudioEncoder(this.MediaId, audioInfo);
        }

        public void Open(string uri, bool bVideo, bool bAudio)
        {
            IsOpen = OeipHelper.openWriteMedia(MediaId, uri, bVideo, bAudio) >= 0;
        }

        public void Close()
        {
            IsOpen = false;
            OeipHelper.closeWriteMedia(MediaId);
        }

        private void OnOperateHandle(int type, int code)
        {
            if (code < 0)
            {
                return;
            }
        }

        public void PushVideoFrame(ref OeipVideoFrame videoFrame)
        {
            if (!IsOpen)
                return;           
            OeipHelper.pushVideo(MediaId, ref videoFrame);
        }

        public void PushAudioFrame(ref OeipAudioFrame audioFrame)
        {
            if (!IsOpen)
                return;           
            OeipHelper.pushAudio(MediaId, ref audioFrame);
        }
    }
}
