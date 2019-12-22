#include "OutputLayerDx11.h"


OutputLayerDx11::OutputLayerDx11() {
	shardTexs.resize(inCount);
	outBuffers.resize(inCount);
	outTexs.resize(inCount);
	for (int32_t i = 0; i < inCount; i++) {
		shardTexs[i] = std::make_unique<Dx11SharedTex>();
		outBuffers[i] = std::make_unique<Dx11Buffer>();
	}
	cpuReadBuffer.resize(inCount);
	computeShader = std::make_unique<Dx11ComputeShader>();
	computeShader->setCS(107, modeName, rctype);
}

bool OutputLayerDx11::onInitBuffer() {
	for (int32_t i = 0; i < inCount; i++) {
		if (layerParamet.bCpu) {
			if (selfConnects[i].dataType == OEIP_CV_8UC4) {
				outBuffers[i]->setOnlyUAV(true);
				outBuffers[i]->setBufferSize(4, selfConnects[i].width * selfConnects[i].height);
				outBuffers[i]->initResource(dx11->device);
				cpuReadBuffer[i].Release();
				copyBufferToRead(dx11->device, outBuffers[i]->buffer, &cpuReadBuffer[i]);
			}
		}
		if (layerParamet.bGpu) {
			for (int32_t i = 0; i < inCount; i++) {
				DXGI_FORMAT dxFormat = getDxFormat(selfConnects[i].dataType);
				shardTexs[i]->restart(dx11->device, selfConnects[i].width, selfConnects[i].height, dxFormat);
				int32_t layerIndex = forwardLayerIndexs[i];
				int32_t inIndex = forwardOutIndexs[i];
				std::shared_ptr<Dx11Texture> texture;
				dx11->getTexture(layerIndex, texture, inIndex);
				outTexs[i] = texture->texture;
			}
		}
	}
	return LayerDx11::onInitBuffer();
}

void OutputLayerDx11::onParametChange(OutputParamet oldParamet) {
	if (!bBufferInit)
		return;
	dx11->resetLayers();
}

bool OutputLayerDx11::initHlsl() {
	return computeShader->initResource(dx11->device, nullptr, dx11->includeShader);
}

void OutputLayerDx11::outputGpuTex(void* device, void* texture, int32_t outputIndex) {
	ID3D11Device* dxdevice = (ID3D11Device*)device;
	ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)texture;
	if (!layerParamet.bGpu || dxdevice == nullptr || dxtexture == nullptr)
		return;
	if (shardTexs[outputIndex] && shardTexs[outputIndex]->bGpuUpdate) {
		//把当前的GPU纹理复制到另一DX上下文的纹理中
		copySharedToTexture(dxdevice, shardTexs[outputIndex]->sharedHandle, dxtexture);
		shardTexs[outputIndex]->bGpuUpdate = false;
	}
}

void OutputLayerDx11::onRunLayer() {
	for (int32_t i = 0; i < outCount; i++) {
		if (layerParamet.bCpu) {
			computeShader->runCS(dx11->ctx, groupSize, inSRVs[i], outBuffers[i]->uavView, constBuffer->buffer);
			dx11->ctx->CopyResource(cpuReadBuffer[i], outBuffers[i]->buffer);
			uint8_t* cpuData = nullptr;
			uint32_t widthByteSize = 0;
			downloadDx11Resource(dx11->ctx, cpuReadBuffer[i], &cpuData, widthByteSize);
			dx11->outputData(layerIndex, cpuData, selfConnects[i].width, selfConnects[i].height, i);
		}
		if (layerParamet.bGpu) {
			if (shardTexs[i]->texture == nullptr)
				return;
			CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
			HRESULT hResult = shardTexs[i]->texture->texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (LPVOID*)&pDX11Mutex);
			DWORD result = pDX11Mutex->AcquireSync(0, 0);
			if (result == WAIT_OBJECT_0) {
				dx11->ctx->CopyResource(shardTexs[i]->texture->texture, outTexs[i]);
				shardTexs[i]->bGpuUpdate = true;
			}
			result = pDX11Mutex->ReleaseSync(1);
		}
	}
}
