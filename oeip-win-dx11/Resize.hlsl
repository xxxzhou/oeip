#include "Common.hlsl"

#ifndef OEIP_LINE_SAMPLER
#define OEIP_LINE_SAMPLER 1
#endif

SamplerState resizeSampler
{
#if (OEIP_LINE_SAMPLER == 1)
	Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;//D3D11_FILTER_MIN_MAG_MIP_POINT
#else
	Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
#endif
	AddressU = Clamp;
	AddressV = Clamp;
};

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};

Texture2D<unorm float4> texIn : register(t0);
RWTexture2D<unorm float4> texOut : register(u0);

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
	if (DTid.x >= width || DTid.y >= height)
		return;
	float2 uv = float2((DTid.x + 0.5) / width, (DTid.y + 0.5) / height);
	texOut[DTid.xy] = texIn.SampleLevel(resizeSampler, uv, 0);
}