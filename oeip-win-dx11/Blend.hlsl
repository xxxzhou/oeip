#include "Common.hlsl"

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
	//float left;
	//float top;
	//float width2;
	//float height2;
	float centerX = 0.f;
	float centerY = 0.f;
	float width2 = 0.f;
	float height2 = 0.f;
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
//Blend
[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
	if (DTid.x >= width || DTid.y >= height)
		return;
	float4 color = texIn[DTid.xy];
	//DTid 线程转化成对应UV坐标，需要分别加0.5
	float2 uv = float2((DTid.x + 0.5) / width, (DTid.y + 0.5) / height);
	float4 rect = float4(centerX - width2 / 2.f, centerY - height2 / 2.f, centerX + width2 / 2.f, centerY + height2 / 2.f);
	if (uv.x >= rect.x && uv.x < rect.z && uv.y >= rect.y && uv.y < rect.w) {
		float2 uv2 = float2((uv.x - rect.x) / width2, (uv.y - rect.y) / height2);
		float4 color2 = texIn2.SampleLevel(lineSampler, uv2, 0);
		color = color2 * (1.f - opacity) + color * opacity;
	}
	texOut[DTid.xy] = color;
}