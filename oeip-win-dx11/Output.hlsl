#include "Common.hlsl"

#ifndef OEIP_DATA_TYPE
#define OEIP_DATA_TYPE 0
#endif

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};

#if (OEIP_DATA_TYPE == 0)
Texture2D<unorm float> texIn : register(t0);
#elif (OEIP_DATA_TYPE == 24)
Texture2D<unorm float4> texIn : register(t0);
#endif

//RWStructuredBuffer可以读写数据,而RWTexture2D用于读的限制非常大,一般不用RWTexture2D用于读
RWStructuredBuffer<uint> dataOut : register(u0);
//Output
//https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math
[numthreads(SIZE_X, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {// uint GI : SV_GroupIndex)	
	if (DTid.x >= width * height)
		return;
	uint2 uv = u12u2(DTid.x, width);
#if (OEIP_DATA_TYPE == 0)	
	float4 rgba = 0.f;
	[unroll]
	for (int i = 0; i < 4; i++) {
		rgba[i] = texIn[uint2(uv.x * 4 + i, uv.y)];
	}
	uint4 data = uint4(rgba * 255);
	dataOut[DTid.x] = uint8Zip(data);
#elif(OEIP_DATA_TYPE == 24)	
	float4 rgba = texIn[uv];
	uint4 data = uint4(rgba * 255);
	dataOut[DTid.x] = uint8Zip(data);
#endif
}
