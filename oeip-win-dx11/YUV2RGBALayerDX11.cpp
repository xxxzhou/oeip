#include "YUV2RGBALayerDX11.h"

YUV2RGBALayerDX11::YUV2RGBALayerDX11()
{
	initConnect();
	computeShader = std::make_unique<Dx11ComputeShader>();
	computeShader->setCS(108, modeName, rctype);
}

void YUV2RGBALayerDX11::initConnect()
{
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP || layerParamet.yuvType == OEIP_YUVFMT_YUY2P || layerParamet.yuvType == OEIP_YUVFMT_YUV420P) {
		selfConnects[0].dataType = OEIP_CV_8UC1;
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
	else {
		selfConnects[0].dataType = OEIP_CV_8UC4;
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
}

void YUV2RGBALayerDX11::onParametChange(YUV2RGBAParamet oldT)
{
	initConnect();
	if (!bBufferInit)
		return;
	dx11->resetLayers();
}

bool YUV2RGBALayerDX11::initHlsl()
{
	std::string yuvTypestr = std::to_string(layerParamet.yuvType);
	// "SIZE_X", OEIP_CS_SIZE_XSTR, "SIZE_Y",OEIP_CS_SIZE_YSTR
	D3D_SHADER_MACRO defines[] = { "OEIP_YUV_TYPE",yuvTypestr.c_str(),nullptr,nullptr };
	return computeShader->initResource(dx11->device, defines, dx11->includeShader);
}

void YUV2RGBALayerDX11::onInitLayer()
{
	if (layerParamet.yuvType == OEIP_YUVFMT_YUV420SP) {
		threadSizeX = selfConnects[0].width;
		threadSizeY = selfConnects[0].height * 2 / 3;
		outConnects[0].width = threadSizeX;
		outConnects[0].height = threadSizeY;
	}
	else if (layerParamet.yuvType == OEIP_YUVFMT_YUY2I || layerParamet.yuvType == OEIP_YUVFMT_YVYUI || layerParamet.yuvType == OEIP_YUVFMT_UYVYI) {
		//这几种模式一个点包含二个Y,一个U,一个V，可以分成二个RGBA元素,还是为了避免访问纹理显存指针冲突
		threadSizeX = selfConnects[0].width;
		threadSizeY = selfConnects[0].height;
		outConnects[0].width = threadSizeX * 2;
		outConnects[0].height = threadSizeY;
	}
	groupSize.X = divUp(threadSizeX, sizeX);
	groupSize.Y = divUp(threadSizeY, sizeY);
	groupSize.Z = 1;
	LayerDx11::onInitLayer();
}

