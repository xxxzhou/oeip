#include "Common.hlsl"

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
	int flipX;
	int flipY;
	float gamma;
};

Texture2D<unorm float4> texIn : register(t0);
RWTexture2D<unorm float4> texOut : register(u0);

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {
	if (DTid.x >= width || DTid.y >= height)
		return;
	uint2 uv = DTid.xy;
	if (flipX == 1) {
		uv.x = width - DTid.x;
	}
	if (flipY == 1) {
		uv.y = height - DTid.y;
	}
	float4 color = texIn[uv];
	float4 rgba = pow(color, gamma);
	texOut[DTid.xy] = rgba;
}