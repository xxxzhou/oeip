#include "Dx11ComputeShader.h"
#include "d3dcommon.h"

void Dx11ComputeShader::setCS(int32_t rcId, std::string modelName, std::string rcType) {
	readResouce(modelName.c_str(), rcId, rcType.c_str(), rcData, size);
}

bool Dx11ComputeShader::initResource(ID3D11Device* deviceDx11, const D3D_SHADER_MACRO* defines, ID3DInclude* include, std::string mainFunc) {
	if (rcData.empty())
		return false;
	computeShader.Release();
	DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(DEBUG) || defined(_DEBUG) 
	dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif
	//const D3D_SHADER_MACRO defines[] = { "USE_STRUCTURED_BUFFERS", "1", nullptr,nullptr };
	LPCSTR pProfile = (deviceDx11->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0) ? "cs_5_0" : "cs_4_0";

	CComPtr<ID3DBlob> pErrorBlob = nullptr;
	CComPtr<ID3DBlob> pBlob = nullptr;

	HRESULT hr = D3DCompile(rcData.c_str(), size, nullptr, defines, include, mainFunc.c_str(), pProfile,
		dwShaderFlags, 0, &pBlob, &pErrorBlob);

	if (FAILED(hr)) {
		if (pErrorBlob) {
			logMessage(OEIP_ERROR, (char*)pErrorBlob->GetBufferPointer());
		}
		return false;
	}
	hr = deviceDx11->CreateComputeShader(pBlob->GetBufferPointer(), pBlob->GetBufferSize(), nullptr, &computeShader);
	return SUCCEEDED(hr);
}

void Dx11ComputeShader::runCS(ID3D11DeviceContext* dxCtx, const UInt3& groupSize, std::vector<ID3D11ShaderResourceView*> srvs, std::vector<ID3D11UnorderedAccessView*> urvs, std::vector<ID3D11Buffer*> cbuffer, bool bClear) {
	dxCtx->CSSetShader(computeShader, nullptr, 0);
	if (cbuffer.size() > 0)
		dxCtx->CSSetConstantBuffers(0, cbuffer.size(), cbuffer.data());
	if (srvs.size() > 0)
		dxCtx->CSSetShaderResources(0, srvs.size(), srvs.data());
	if (urvs.size() > 0)
		dxCtx->CSSetUnorderedAccessViews(0, urvs.size(), urvs.data(), nullptr);
	dxCtx->Dispatch(groupSize.X, groupSize.Y, groupSize.Z);
	dxCtx->CSSetShader(nullptr, nullptr, 0);
	if (bClear) {
		if (cbuffer.size() > 0) {
			std::vector<ID3D11Buffer*> ppCBnullptr(cbuffer.size(), nullptr);
			dxCtx->CSSetConstantBuffers(0, ppCBnullptr.size(), ppCBnullptr.data());
		}
		if (srvs.size() > 0) {
			std::vector<ID3D11ShaderResourceView*> ppSRVnullptr(srvs.size(), nullptr);
			dxCtx->CSSetShaderResources(0, ppSRVnullptr.size(), ppSRVnullptr.data());
		}
		if (urvs.size() > 0) {
			std::vector<ID3D11UnorderedAccessView*> ppUAViewnullptr(urvs.size(), nullptr);
			dxCtx->CSSetUnorderedAccessViews(0, ppUAViewnullptr.size(), ppUAViewnullptr.data(), nullptr);
		}
	}
}

void Dx11ComputeShader::runCS(ID3D11DeviceContext* dxCtx, const UInt3& groupSize, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* urv, ID3D11Buffer* cbuffer, bool bClear) {
	dxCtx->CSSetShader(computeShader, nullptr, 0);
	if (cbuffer)
		dxCtx->CSSetConstantBuffers(0, 1, &cbuffer);
	if (srv)
		dxCtx->CSSetShaderResources(0, 1, &srv);
	if (urv)
		dxCtx->CSSetUnorderedAccessViews(0, 1, &urv, nullptr);
	dxCtx->Dispatch(groupSize.X, groupSize.Y, groupSize.Z);
	dxCtx->CSSetShader(nullptr, nullptr, 0);
	if (bClear) {
		if (cbuffer) {
			ID3D11Buffer* pbuffer = nullptr;
			dxCtx->CSSetConstantBuffers(0, 1, &pbuffer);
		}
		if (srv) {
			ID3D11ShaderResourceView* pbuffer = nullptr;
			dxCtx->CSSetShaderResources(0, 1, &pbuffer);
		}
		if (urv) {
			ID3D11UnorderedAccessView* pbuffer = nullptr;
			dxCtx->CSSetUnorderedAccessViews(0, 1, &pbuffer, nullptr);
		}
	}
}

