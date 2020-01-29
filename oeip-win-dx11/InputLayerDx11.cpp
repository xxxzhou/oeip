#include "InputLayerDx11.h"


InputLayerDx11::InputLayerDx11() {
	shardTexs.resize(inCount);
	inBuffers.resize(inCount);
	cpuUpdates.resize(inCount, false);
	for (int32_t i = 0; i < inCount; i++) {
		shardTexs[i] = std::make_unique< Dx11SharedTex>();
		inBuffers[i] = std::make_unique< Dx11Buffer>();
	}
	computeShader = std::make_unique<Dx11ComputeShader>();
	computeShader->setCS(105, modeName, rctype);
}

void InputLayerDx11::onParametChange(InputParamet oldT) {
	if (!bBufferInit)
		return;
	dx11->resetLayers();
}

void InputLayerDx11::onInitLayer() {
	LayerDx11::onInitLayer();
	//因StructureByteStride只能为4的倍数,相应buffer要重新设计,图片的宽度不为4的倍数后面想办法处理
	if (selfConnects[0].dataType == OEIP_CV_8UC1) {
		threadSizeX = divUp(threadSizeX, 4);
	}
	else if (selfConnects[0].dataType == OEIP_CV_8UC3) {
		//OEIP_CV_8UC3经过运算转成OEIP_CV_8UC4,DX11中没有适配OEIP_CV_8UC3的格式
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
	//threadSizeX在这表示有多少个像素点，每个像素点可能是C1/C3/C4
	//240=3*4*5*4,8UC3为免访问SB显存指针冲突，用共享显存来处理,3的倍数块可以只访问本块共享显存
	//同时也避免一些CPU数据直接上传到纹理要求的一些宽度对齐,如32倍数这些
	groupSize.X = divUp(threadSizeX * threadSizeY, 240);
	groupSize.Y = 1;
	groupSize.Z = 1;
}
//OEIP_CV_8UC3 没有相应的纹理格式
bool InputLayerDx11::onInitBuffer() {
	bool bInit = true;
	for (int32_t i = 0; i < inCount; i++) {
		DXGI_FORMAT dxFormat = getDxFormat(selfConnects[i].dataType);
		if (layerParamet.bCpu) {
			inBuffers[i]->setCpuWrite(true);
			//StructureByteStride最小为4
			if (selfConnects[i].dataType == OEIP_CV_8UC1) {
				inBuffers[i]->setBufferSize(4, selfConnects[i].width * selfConnects[i].height / 4);
			}
			else if (selfConnects[i].dataType == OEIP_CV_8UC3) {
				inBuffers[i]->setBufferSize(4, selfConnects[i].width * selfConnects[i].height * 3 / 4);
			}
			else if (selfConnects[i].dataType == OEIP_CV_8UC4) {
				inBuffers[i]->setBufferSize(4, selfConnects[i].width * selfConnects[i].height);
			}
			inBuffers[i]->initResource(dx11->device);
		}
		if (layerParamet.bGpu) {
			if (selfConnects[i].dataType == OEIP_CV_8UC3)//OEIP_CV_8UC3
				dxFormat = DXGI_FORMAT_B8G8R8A8_UNORM;
			shardTexs[i]->restart(dx11->device, selfConnects[i].width, selfConnects[i].height, dxFormat);
		}		
		outTextures[i]->setTextureSize(outConnects[i].width, outConnects[i].height, dxFormat);
		bInit &= outTextures[i]->initResource(dx11->device);
		outUAVs[i] = outTextures[i]->uavView;
	}
	onInitCBuffer();
	updateCBuffer();
	return true;
	//return LayerDx11::onInitBuffer();
}

bool InputLayerDx11::initHlsl() {
	std::string yuvTypestr = std::to_string(selfConnects[0].dataType);
	D3D_SHADER_MACRO defines[] = { "OEIP_DATA_TYPE",yuvTypestr.c_str(),"SIZE_X", "240", "SIZE_Y","1",nullptr,nullptr };
	return computeShader->initResource(dx11->device, defines, dx11->includeShader);
}

void InputLayerDx11::onRunLayer() {
	for (int32_t i = 0; i < inCount; i++) {
		if (layerParamet.bCpu && cpuUpdates[i]) {
			inBuffers[i]->updateResource(dx11->ctx);
			computeShader->runCS(dx11->ctx, groupSize, inBuffers[i]->srvView, outTextures[i]->uavView, constBuffer->buffer);
			cpuUpdates[i] = false;
		}
		if (layerParamet.bGpu) {
			if (shardTexs[i]->texture == nullptr || !shardTexs[i]->bGpuUpdate)
				return;
			CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
			HRESULT hResult = shardTexs[0]->texture->texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (LPVOID*)&pDX11Mutex);
			DWORD result = pDX11Mutex->AcquireSync(0, 0);
			if (result == WAIT_OBJECT_0) {
				dx11->ctx->CopyResource(outTextures[i]->texture, shardTexs[i]->texture->texture);
				shardTexs[i]->bGpuUpdate = false;
			}
			result = pDX11Mutex->ReleaseSync(1);
		}
	}
}

void InputLayerDx11::inputCpuData(uint8_t* byteData, int32_t inputIndex) {
	//CPU更新
	inBuffers[inputIndex]->cpuData = byteData;
	cpuUpdates[inputIndex] = true;
}

void InputLayerDx11::inputGpuTex(void* device, void* texture, int32_t inputIndex) {
	ID3D11Device* dxdevice = (ID3D11Device*)device;
	ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)texture;
	if (!layerParamet.bGpu || dxdevice == nullptr || dxtexture == nullptr)
		return;
	//把别的ctx的GPU纹理复制到当前GPU纹理上
	copyTextureToShared(dxdevice, shardTexs[inputIndex]->sharedHandle, dxtexture);
	shardTexs[inputIndex]->bGpuUpdate = true;
}
