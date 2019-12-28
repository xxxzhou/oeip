#pragma once
#include "../oeip-win/Dx11Helper.h"
#include "../oeip-win/Dx11Resource.h"

struct UInt3
{
	uint32_t X = 1;
	uint32_t Y = 1;
	uint32_t Z = 1;

	UInt3(uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) {
		X = x;
		Y = y;
		Z = z;
	}
};

class Dx11ComputeShader
{
public:
	Dx11ComputeShader() {};
	virtual ~Dx11ComputeShader() {};
private:
	CComPtr<ID3D11ComputeShader> computeShader = nullptr;
	std::string rcData = "";
	uint32_t size = 0;
public:
	void setCS(int32_t rcId, std::string modelName, std::string rcType);
	bool initResource(ID3D11Device* deviceDx11, const D3D_SHADER_MACRO* macro, ID3DInclude* include, std::string mainFunc = "main");
	void runCS(ID3D11DeviceContext* dxCtx, const UInt3& groupSize, std::vector<ID3D11ShaderResourceView*> srvs,
		std::vector<ID3D11UnorderedAccessView*> uavs, std::vector<ID3D11Buffer*> cbuffer, bool bClear = true);
	void runCS(ID3D11DeviceContext* dxCtx, const UInt3& groupSize, ID3D11ShaderResourceView* srv, ID3D11UnorderedAccessView* uav, ID3D11Buffer* cbuffer, bool bClear = true);
};

