using OeipCommon;
using OeipWrapper;
using OeipWrapper.FixPipe;
using OeipWrapper.Live;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class TestLive : MonoBehaviour
{
    public bool bPullSelf;
    public Text label;
    public RectTransform cameraSelectPanel;
    public RectTransform grabCutPanel;
    public MeshRenderer showPanel;
    public MeshRenderer livePanel;
    public Button btnLoadNet;
    public Button btnGrabcut;
    public CameraView cameraView;
    private PersonBox[] personBox = null;
    private bool bDrawMode = false;
    private Setting setting;
    private DarknetParamet darknetParamet = new DarknetParamet();

    private ObjectBind3D<CameraSetting> objBindCamera = new ObjectBind3D<CameraSetting>();
    private ObjectBind3D<OeipVideoParamet> objBindGrabcut = new ObjectBind3D<OeipVideoParamet>();

    public LiveView liveView;
    public Button btnLoginRoom;
    public Button btnLogoutRoom;
    public InputField inputRoom;
    public InputField inputUserId;
    // Start is called before the first frame update
    void Start()
    {
        setting = SettingManager.Instance.Setting;
        OeipManager.Instance.OnLogEvent += Instance_OnLogEvent;
        //绑定Camera UI       
        objBindCamera.Bind(setting.cameraSetting, cameraSelectPanel);
        objBindCamera.GetComponent<DropdownComponent>("CameraIndex").SetFillOptions(true, OeipManagerU3D.Instance.GetCameras);
        objBindCamera.GetComponent<DropdownComponent>("FormatIndex").SetFillOptions(true, OeipManagerU3D.Instance.GetFormats);
        objBindCamera.OnChangeEvent += ObjBindCamera_OnChangeEvent;
        //cameraView管线返回设置
        cameraView.VideoPipe.Pipe.OnProcessEvent += Pipe_OnProcessEvent;
        //绑定GrabCut设置
        objBindGrabcut.Bind(setting.videoParamet, grabCutPanel);
        cameraView.VideoPipe.UpdateVideoParamet(setting.videoParamet);
        objBindGrabcut.OnChangeEvent += ObjBindGrabcut_OnChangeEvent;
        //加载神经网络
        btnLoadNet.onClick.AddListener(OnLoadNet);
        //Grabcut 扣像
        btnGrabcut.onClick.AddListener(OnGrabCut);
        //加载Live
        OeipLiveManager.Instance.OnLoginRoomEvent += Instance_OnLoginRoomEvent;
        OeipLiveManager.Instance.OnStreamUpdateEvent += Instance_OnStreamUpdateEvent;
        btnLoginRoom.onClick.AddListener(OnLoginRoom);
        btnLogoutRoom.onClick.AddListener(OnLogoutRoom);
        liveView.OnLiveTexChange += LiveView_OnLiveTexChange;
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
    }

    private void Instance_OnStreamUpdateEvent(int userId, int index, bool bAdd)
    {
        if (!bPullSelf && OeipLiveManager.Instance.UserId == userId)
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

    private void ObjBindGrabcut_OnChangeEvent(OeipCommon.OeipAttribute.ObjectBind<OeipVideoParamet> arg1, string arg2)
    {
        //更新grab cut扣像参数
        cameraView.VideoPipe.UpdateVideoParamet(arg1.Obj);
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

    private void ObjBindCamera_OnChangeEvent(OeipCommon.OeipAttribute.ObjectBind<CameraSetting> arg1, string arg2)
    {
        cameraView.OpenCamera(setting.cameraSetting.CameraIndex, setting.cameraSetting.FormatIndex);
        showPanel.material.SetTexture("_MainTex", cameraView.SourceTex);
    }

    private void OnDestroy()
    {
        OeipLiveManager.Instance.Close();
        OeipManager.Instance.Close();
        SettingManager.Instance.Close();
    }

    private void OeipCamera_OnReviceEvent(IntPtr data, int width, int height)
    {
        //摄像机输入数据
        cameraView.VideoPipe.RunVideoPipe(data);
    }

    private void Pipe_OnProcessEvent(int layerIndex, IntPtr data, int width, int height, int outputIndex)
    {
        if (layerIndex == cameraView.VideoPipe.OutYuvIndex)
        {
            OeipLiveManager.Instance.PushVideoFrame(0, data, width, height, cameraView.VideoPipe.YUVFMT);
        }
        else if (layerIndex == cameraView.VideoPipe.DarknetIndex)
        {
            if (width > 0)
            {
                personBox = PInvokeHelper.GetPInvokeArray<PersonBox>(width, data);
                Loom.QueueOnMainThread(() =>
                {
                    if (personBox == null)
                        return;
                    string msg = string.Empty;
                    foreach (var px in personBox)
                    {
                        msg += " " + px.prob;
                    }
                    label.text = $"人数:{width} {msg}";
                });
            }
        }
    }

    private void OnLoadNet()
    {
        darknetParamet.bLoad = 1;
        darknetParamet.confile = Path.GetFullPath(Path.Combine(Application.dataPath, "../../ThirdParty/yolov3-tiny-test.cfg"));
        darknetParamet.weightfile = Path.GetFullPath(Path.Combine(Application.dataPath, "../../ThirdParty/yolov3-tiny_745000.weights"));
        darknetParamet.thresh = 0.3f;
        darknetParamet.nms = 0.3f;
        darknetParamet.bDraw = 1;
        darknetParamet.drawColor = OeipHelper.getColor(1.0f, 0.1f, 0.1f, 0.8f);
        Loom.RunAsync(() =>
        {
            cameraView.VideoPipe.UpdateDarknetParamet(ref darknetParamet);
        });
    }

    private void OnGrabCut()
    {
        Loom.RunAsync(() =>
        {
            bDrawMode = !bDrawMode;
            OeipRect rect = new OeipRect();
            if (personBox != null && personBox.Length > 0)
            {
                rect = personBox[0].rect;
            }
            cameraView.VideoPipe.ChangeGrabcutMode(bDrawMode, ref rect);
        });
    }

    private void OnLoginRoom()
    {
        int userId = 21;
        int.TryParse(inputUserId.text, out userId);
        OeipLiveManager.Instance.LoginRoom(inputRoom.text, userId);
    }

    private void OnLogoutRoom()
    {
        OeipLiveManager.Instance.LogoutRoom();
    }
}
