#include "Common.hlsl"

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};

Texture2D<unorm float4> texIn : register(t0);
//RWStructuredBuffer可以读写数据,而RWTexture2D用于读的限制非常大,一般不用RWTexture2D用于读
RWStructuredBuffer<uint> dataOut : register(u0);

//https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math
[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)// uint GI : SV_GroupIndex)
{
	if (DTid.x >= width || DTid.y >= height)
		return;
	uint index = u22u1(DTid.xy, width);
	float4 rgba = texIn[DTid.xy];
	uint4 data = uint4(rgba * 255);
	dataOut[index] = uint8Zip(data);	
}                                  
