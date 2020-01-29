using OeipCommon;
using OeipWrapper;
using OeipWrapper.Live;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TestLive2 : MonoBehaviour
{
    public MeshRenderer livePanel;

    public RenderTexture renderTexture;
    public LiveView liveView;
    public LivePushPipe livePushPipe;
    public Transform moveObj;

    public Button btnLoginRoom;
    public Button btnLogoutRoom;
    public InputField inputRoom;
    public InputField inputUserId;

    private Vector3 opos;
    // Start is called before the first frame update
    void Start()
    {
        livePushPipe.SetRenderTarget(renderTexture);
        livePushPipe.pipe.OnProcessEvent += Pipe_OnProcessEvent;

        btnLoginRoom.onClick.AddListener(OnLoginRoom);
        btnLogoutRoom.onClick.AddListener(OnLogoutRoom);

        OeipLiveManager.Instance.OnLoginRoomEvent += Instance_OnLoginRoomEvent;
        OeipLiveManager.Instance.OnStreamUpdateEvent += Instance_OnStreamUpdateEvent;

        liveView.OnLiveTexChange += LiveView_OnLiveTexChange;

        opos = moveObj.localPosition;
    }

    private void LiveView_OnLiveTexChange()
    {
        livePanel.material.SetTexture("_MainTex", liveView.SourceTex);
    }

    //登陆成功后推流
    private void Instance_OnLoginRoomEvent(int code, int userid)
    {
        OeipPushSetting pushSetting = new OeipPushSetting();
        pushSetting.bVideo = 1;
        pushSetting.bAudio = 0;
        OeipLiveManager.Instance.PushStream(0, ref pushSetting);
        livePushPipe.SetPush(true);
    }

    private void Instance_OnStreamUpdateEvent(int userId, int index, bool bAdd)
    {
        if (OeipLiveManager.Instance.UserId == userId)
            return;
        if (bAdd)
        {
            OeipLiveManager.Instance.PullStream(userId, index);
            liveView.SetPullUserIndex(userId, index);
        }
        else
        {
            OeipLiveManager.Instance.StopPullStream(userId, index);
            liveView.SetPullUserIndex(-1, -1);
        }
    }

    private void Instance_OnLogEvent(int level, string message)
    {
        OeipLogLevel logLevel = (OeipLogLevel)level;
        if (logLevel == OeipLogLevel.OEIP_INFO)
        {
            Debug.Log(message);
        }
        else if (logLevel >= OeipLogLevel.OEIP_WARN)
        {
            Debug.LogWarning(message);
        }
    }

    private void Pipe_OnProcessEvent(int layerIndex, System.IntPtr data, int width, int height, int outputIndex)
    {
        if (layerIndex == livePushPipe.OutputIndex)
        {
            OeipLiveManager.Instance.PushVideoFrame(0, data, width, height, livePushPipe.yUVFMT);
        }
    }

    // Update is called once per frame
    void Update()
    {
        moveObj.localPosition = opos + Vector3.up * Mathf.Sin(Time.time);       
    }

    private void OnLoginRoom()
    {
        int userId = 21;
        int.TryParse(inputUserId.text, out userId);
        OeipLiveManager.Instance.LoginRoom(inputRoom.text, userId);
    }

    private void OnLogoutRoom()
    {
        livePushPipe.SetPush(false);
        OeipLiveManager.Instance.LogoutRoom();
    }


    private void OnDestroy()
    {
        OeipLiveManager.Instance.Close();
        OeipManager.Instance.Close();
    }
}
