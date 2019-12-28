#include "Common.hlsl"

#ifndef OEIP_YUV_TYPE
#define OEIP_YUV_TYPE 2
#endif

cbuffer texSize : register(b0)
{
	uint width;
	uint height;
	uint elementCount;
	uint elementByte;
};

//1 OEIP_YUVFMT_YUV420SP NV12
//2 OEIP_YUVFMT_YUV2
//3 OEIP_YUVFMT_YVYUI
//4 OEIP_YUVFMT_UYVYI
//5 OEIP_YUVFMT_YUY2P
//6 OEIP_YUVFMT_YUV420P

Texture2D<unorm float4> texIn : register(t0);

#if (OEIP_YUV_TYPE == 1 || OEIP_YUV_TYPE == 5 || OEIP_YUV_TYPE == 6)
RWTexture2D<unorm float> texOut : register(u0);
#elif (OEIP_YUV_TYPE == 2 || OEIP_YUV_TYPE == 3 || OEIP_YUV_TYPE == 4)
RWTexture2D<unorm float4> texOut : register(u0);
#endif

[numthreads(SIZE_X, SIZE_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID) {//uint GI : SV_GroupIndex

	if (DTid.x >= width || DTid.y >= height)
		return;
#if (OEIP_YUV_TYPE == 1 || OEIP_YUV_TYPE == 5 || OEIP_YUV_TYPE == 6)
	float4 rgba = texIn[DTid.xy];
	float4 yuv = rgb2Yuv(rgba);
	uint2 yIndex = DTid.xy;
#if (OEIP_YUV_TYPE == 1)//OEIP_YUVFMT_YUV420SP
	uint2 uvv = uint2((DTid.x >> 1) << 1, DTid.y >> 1);
	uint2 uIndex = uint2(0, height) + uvv;
	uint2 vIndex = uIndex + uint2(1, 0);
#elif (OEIP_YUV_TYPE == 5)//OEIP_YUVFMT_YUY2P	
	uint2 uIndex = uint2(0, height) + uint2(DTid.x, DTid.y >> 1);
	uint2 vIndex = uint2(0, height * 3 / 2) + uint2(DTid.x, DTid.y >> 1);
#else//OEIP_YUVFMT_YUV420P
	uint2 uIndex = uint2(0, height) + uint2(DTid.x, DTid.y >> 2);
	uint2 vIndex = uint2(0, height * 5 / 4) + uint2(DTid.x, DTid.y >> 2);
#endif
	texOut[yIndex] = yuv.x;
	texOut[uIndex] = yuv.y;
	texOut[vIndex] = yuv.z;
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
	float4 rgba1 = texIn[uint2(DTid.x * 2, DTid.y)];
	float4 rgba2 = texIn[uint2(DTid.x * 2 + 1, DTid.y)];
	float4 yuv1 = rgb2Yuv(rgba1);
	float4 yuv2 = rgb2Yuv(rgba2);
	float4 yuyv = float4(yuv1.x, (yuv1.y + yuv2.y) / 2.f, yuv2.x, (yuv1.z + yuv2.z) / 2.f);
	float4 syuyv = float4(yuyv[yoffset], yuyv[bitx + (1 - yoffset)], yuyv[yoffset + 2], yuyv[(2 - bitx) + (1 - yoffset)]);
	texOut[uint2(DTid.x, DTid.y)] = syuyv;
#endif
}