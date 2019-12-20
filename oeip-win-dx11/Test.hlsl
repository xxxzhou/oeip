#include "Common.hlsl"

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};
Texture2D<unorm float4> texIn : register(t0);
RWStructuredBuffer<uint> dataOut : register(u0);

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
	if (DTid.x >= width || DTid.y >= height)
		return;
	uint index = u22u1(DTid.xy, width);
	float4 rgba = texIn[DTid.xy];
	uint4 data = uint4(rgba * 255);
	dataOut[index] = data.x | data.y << 8 | data.z << 16 | data.w << 24;
}