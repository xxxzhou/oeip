# oeip
多媒体与游戏引擎(主要支持Unity3D/UE4)
详细介绍 https://zhuanlan.zhihu.com/p/104027165

编译项目:
CUDA 10.1安裝:
https://developer.nvidia.com/cuda-downloads
CUDNN 10.1安裝:
https://developer.nvidia.com/cudnn

下载 https://github.com/xxxzhou/oeip-thridparty
在Oeip项目下，新建一个ThirdParty文件夹，把oeip-thridparty里的文件全部复制到这。
二种引用DLL方式。
一是把相应的DLL复制到对应OEIP dll目录下。
二是在环境变量里把上面的几个文件夹的BIN目录写入

为什么想做这个，主要因为如下几点:
一是现有插件很多都是在插件层拿到CPU数据提交到显存，然后使用游戏引擎本身的上下文处理，这是一个浪费，游戏如在90FPS，而常见多媒体播放只有30FPS。
二GPGPU不同平台不同硬件需要选择不同技术导致的差异不能很好抽象，如DX11使用HLSL+ComputeShader,而N卡上可以使用CUDA，提供一个外层抽象，方便在DX11/CUDA/Vulkun(Vulkun会比较靠后)。
三是与深度学习框架高效结合展示,使用CUDA平台可以高效整合很多深度框架。

设计目标:
1 类似深度学习框架,抽象功能成各个层,每层接收多个输入层与输出层
2 游戏引擎里的纹理数据可以直接复制到我们GPGPU框架里的显存中,包含相应的输入与输出。
3 足够的自由度，用户可以和深度学习框架一样，组装相应逻辑。
4 输入源可以包含游戏窗口，游戏内纹理，摄像头，视频文件，拉流地址
5 Unity3D/UE4的统一接口设计

后续:
1 (已完成)完善层之间逻辑，如某层不启用(一是当前层不启用,二是后续关联层全被关闭) 
2 (已完成)完善层连接逻辑，如输入流NV12/YUV422/ARGB32不同数据流输出相同数据的处理
3 完成一个CUDA初步模块,提供Grabcut算法,darknet集成。(已完成)
4 多媒体直播(推流，拉流等)(已完成)
5 完善C#封装层,C# WinForm相关DEMO代码(已完成)
6 完善UE4/Unity3D 相关DEMO代码
7 传入DX11纹理的测试(已完成)




