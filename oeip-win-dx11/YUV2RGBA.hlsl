#include "Common.hlsl"

#ifndef OEIP_YUV_TYPE
#define OEIP_YUV_TYPE 1
#endif

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};

#if (OEIP_YUV_TYPE == 1 || OEIP_YUV_TYPE == 5 || OEIP_YUV_TYPE == 6)
Texture2D<unorm float> texIn : register(t0);
#elif (OEIP_YUV_TYPE == 2 || OEIP_YUV_TYPE == 3 || OEIP_YUV_TYPE == 4)
Texture2D<unorm float4> texIn : register(t0);
#endif
RWTexture2D<unorm float4> texOut : register(u0);

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)//uint GI : SV_GroupIndex
{
	if (DTid.x >= width || DTid.y >= height)
		return;
	// uint/2*2变成偶数	
	float4 rgba = float4(0, 0, 0, 1);
#if (OEIP_YUV_TYPE == 1 || OEIP_YUV_TYPE == 5 || OEIP_YUV_TYPE == 6)
	uint2 yIndex = DTid.xy;
#if (OEIP_YUV_TYPE == 1)//OEIP_YUVFMT_YUV420SP
	uint2 uvv = uint2((DTid.x >> 1) << 1, DTid.y >> 1);
	uint2 uIndex = uint2(0, height) + uvv;
	uint2 vIndex = uIndex + uint2(1, 0);
#elif (OEIP_YUV_TYPE == 5)//OEIP_YUVFMT_YUY2P	
	uint2 uIndex = uint2(0, height) + uint2(DTid.x >> 1, DTid.y);
	uint2 vIndex = uint2(0, height * 3 / 2) + uint2(DTid.x >> 1, DTid.y);
#else//OEIP_YUVFMT_YUV420P
	uint2 uIndex = uint2(0, height) + uint2(DTid.x >> 2, DTid.y);
	uint2 vIndex = uint2(0, height * 5 / 4) + uint2(DTid.x >> 2, DTid.y);
#endif
	float y = texIn[yIndex];
	float u = texIn[uIndex] - 0.5f;
	float v = texIn[vIndex] - 0.5f;
	rgba = yuv2Rgb(y, u, v, 1.f);
	texOut[DTid.xy] = rgba;
#elif (OEIP_YUV_TYPE == 2 || OEIP_YUV_TYPE == 3 || OEIP_YUV_TYPE == 4)
	//OEIP_YUVFMT_YUV2
	int bitx = 0;
	int yoffset = 0;
#if (OEIP_YUV_TYPE == 3)//OEIP_YUVFMT_YVYUI
	bitx = 2;
#endif
#if (OEIP_YUV_TYPE == 4)//OEIP_YUVFMT_UYVYI
	yoffset = 1;
#endif
	float4 yuyv = texIn[DTid.xy];
	float y1 = yuyv[yoffset];
	float u = yuyv[bitx + (1 - yoffset)] - 0.5f;
	float y2 = yuyv[yoffset + 2];
	float v = yuyv[(2 - bitx) + (1 - yoffset)] - 0.5f;
	texOut[uint2(DTid.x * 2, DTid.y)] = yuv2Rgb(y1, u, v, 1.f);
	texOut[uint2(DTid.x * 2 + 1, DTid.y)] = yuv2Rgb(y2, u, v, 1.f);
#endif
}