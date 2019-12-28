#include "Common.hlsl"

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
	float left;
	float top;
	float width2;
	float height2;
	//不透明度
	float opacity;
};

SamplerState lineSampler
{
	Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;//D3D11_FILTER_MIN_MAG_MIP_POINT
	AddressU = Clamp;
	AddressV = Clamp;
};

Texture2D<unorm float4> texIn : register(t0);
Texture2D<unorm float4> texIn2 : register(t1);
RWTexture2D<unorm float4> texOut : register(u0);

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
	if (DTid.x >= width || DTid.y >= height)
		return;
	float4 color = texIn[DTid.xy];
	float2 uv = float2((DTid.x + 0.5) / width, (DTid.y + 0.5) / height);
	if (uv.x >= left && uv.x < left + width2 && uv.y >= top && uv.y < top + height2) {
		float2 uv2 = float2((uv.x - left) / width2, (uv.y - top) / height2);
		float4 color2 = texIn2.SampleLevel(lineSampler, uv2, 0);
		color = color2 * (1.f - opacity) + color * opacity;
	}
	texOut[DTid.xy] = color;
}