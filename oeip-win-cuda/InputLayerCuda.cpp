#include "InputLayerCuda.h"
#include "cuda_help.h"

void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest, cudaStream_t stream);

InputLayerCuda::InputLayerCuda() {
	shardTexs.resize(inCount);
	cudaResoures.resize(inCount);
	tempMat.resize(inCount);
	cpuUpdates.resize(inCount, false);
	for (int32_t i = 0; i < inCount; i++) {
		shardTexs[i] = std::make_shared< Dx11SharedTex>();
	}
}

InputLayerCuda::~InputLayerCuda() {
}

void InputLayerCuda::onParametChange(InputParamet oldT) {
	ipCuda->resetLayers();
}

void InputLayerCuda::onInitLayer() {
	if (selfConnects[0].dataType == OEIP_CV_8UC3) {
		outConnects[0].dataType = OEIP_CV_8UC4;
	}
}

bool InputLayerCuda::onInitBuffer() {
	if (layerParamet.bGpu) {
		device.Release();
		ctx.Release();
		createDevice11(&device, &ctx);
	}
	for (int32_t i = 0; i < inCount; i++) {
		if (layerParamet.bCpu) {
			if (selfConnects[i].dataType == OEIP_CV_8UC3) {
				tempMat[i].create(selfConnects[i].height, selfConnects[i].width, selfConnects[i].dataType);
			}
		}
		//暂时只支持RGBA
		if (layerParamet.bGpu) {
			DXGI_FORMAT dxFormat = getDxFormat(selfConnects[i].dataType);
			registerCudaResource(cudaResoures[i], shardTexs[i], device, selfConnects[i].width, selfConnects[i].height);
		}
	}
	return LayerCuda::onInitBuffer();
}

void InputLayerCuda::onRunLayer() {
	for (int32_t i = 0; i < inCount; i++) {
		if (layerParamet.bCpu && cpuUpdates[i]) {
			auto frame = cv::Mat(selfConnects[i].height, selfConnects[i].width, selfConnects[i].dataType, cudaResoures[i].cpuData);
			if (selfConnects[i].dataType == OEIP_CV_8UC3) {
				tempMat[i].upload(frame, ipCuda->stream);
				rgb2rgba_gpu(tempMat[i], outMats[i], ipCuda->cudaStream);
			}
			else {
				outMats[i].upload(frame, ipCuda->stream);
			}
			cpuUpdates[i] = false;
		}
		if (layerParamet.bGpu) {
			if (shardTexs[i]->texture == nullptr || !shardTexs[i]->bGpuUpdate)
				return;
			CComPtr<IDXGIKeyedMutex> pDX11Mutex = nullptr;
			HRESULT hResult = shardTexs[0]->texture->texture->QueryInterface(__uuidof(IDXGIKeyedMutex), (LPVOID*)&pDX11Mutex);
			DWORD result = pDX11Mutex->AcquireSync(0, 0);
			if (result == WAIT_OBJECT_0) {
				d3dTexture2GpuMat(outMats[i], cudaResoures[i], ipCuda->cudaStream);
				shardTexs[i]->bGpuUpdate = false;
			}
			result = pDX11Mutex->ReleaseSync(1);
		}
	}
}

void InputLayerCuda::inputGpuTex(void * device, void * texture, int32_t inputIndex) {
	ID3D11Device* dxdevice = (ID3D11Device*)device;
	ID3D11Texture2D* dxtexture = (ID3D11Texture2D*)texture;
	if (!layerParamet.bGpu || dxdevice == nullptr || dxtexture == nullptr)
		return;
	//把别的ctx的GPU纹理复制到当前GPU纹理上
	copyTextureToShared(dxdevice, shardTexs[inputIndex]->sharedHandle, dxtexture);
	shardTexs[inputIndex]->bGpuUpdate = true;
}

void InputLayerCuda::inputCpuData(uint8_t * byteData, int32_t inputIndex) {
	cudaResoures[inputIndex].cpuData = byteData;
	cpuUpdates[inputIndex] = true;
}
