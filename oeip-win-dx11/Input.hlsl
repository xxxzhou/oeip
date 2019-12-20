#include "Common.hlsl"

#ifndef OEIP_DATA_TYPE
#define OEIP_DATA_TYPE 16
#endif

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};
//ByteAddressBuffer
StructuredBuffer<uint> dataIn : register(t0);

#if (OEIP_DATA_TYPE == 0)
RWTexture2D<unorm float> texOut : register(u0);
#elif (OEIP_DATA_TYPE == 16 || OEIP_DATA_TYPE == 24)
RWTexture2D<unorm float4> texOut : register(u0);
#endif

#if (OEIP_DATA_TYPE == 16 )
//避免访问SB显存指针冲突，用共享显存来处理
groupshared uint4 shared_datat[SIZE_X];
#endif

//https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-per-component-math
[numthreads(SIZE_X, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID, uint GI : SV_GroupIndex)
{
#if (OEIP_DATA_TYPE != 16)
	if (DTid.x >= width * height)
		return;
#endif
	uint2 uv = u12u2(DTid.x, width);
	uint data = dataIn[DTid.x];
	uint4 rgba = uint32Unzip(data);
#if (OEIP_DATA_TYPE == 0)
	[unroll]
	for (int i = 0; i < 4; i++) {
		texOut[uint2(uv.x * 4 + i, uv.y)] = clamp(rgba[i] * 0.00392156862745f, 0.f, 1.f);
	}
#elif (OEIP_DATA_TYPE == 16)//OEIP_CV_8UC3
	shared_datat[GI] = rgba;
	//只是当前块的所有处理都同步完成
	GroupMemoryBarrierWithGroupSync();
	if (DTid.x < width * height) {
		uint4 xrgba = uint4(0, 0, 0, 255);
		uint bgi = GI * 3;
		uint si = bgi >> 2;
		uint yi = bgi % 4;
		uint4 mask4 = uint4(0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000);
		[unroll]
		for (int i = 0; i < 3; i++) {
			//新的偏移
			uint yo = (yi + i) >> 2;
			//元素内索引
			uint so = (yi + i) % 4;
			xrgba[i] = (shared_datat[si + yo] & mask4[so]) >> (so * 8);
		}
		texOut[uv] = uchar2float(xrgba);
	}
#elif (OEIP_DATA_TYPE == 24)	
	texOut[uv] = uchar2float(rgba);
#endif
}