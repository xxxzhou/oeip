#include "Common.hlsl"

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
	uint red;
	uint green;
	uint blue;
	uint alpha;
};

Texture2D<unorm float4> texIn : register(t0);
RWTexture2D<unorm float4> texOut : register(u0);

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= width || DTid.y >= height)
		return;
	float4 color = texIn[DTid.xy];
	float4 rgba = float4(color[red], color[green], color[blue], color[alpha]);
	texOut[DTid.xy] = rgba;
}