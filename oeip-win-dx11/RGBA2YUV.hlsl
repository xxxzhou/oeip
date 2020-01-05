#include "Common.hlsl"

#ifndef OEIP_YUV_TYPE
#define OEIP_YUV_TYPE 6
#endif

cbuffer texSize : register(b0)
{
	uint width;//对应线程组宽度
	uint height;//对应线程组长度
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
	//纹理本身会扩展32倍长度，这里不加这个，如果是(1920,1080),height会以1088执行导致结果出错
	if (DTid.x >= width || DTid.y >= height)
		return;
	//OEIP_YUVFMT_YUY2P	
#if (OEIP_YUV_TYPE == 5)
	uint2 uvt = uint2(DTid.x, DTid.y * 2);
	uint2 uvb = uint2(DTid.x, DTid.y * 2 + 1);
	float4 rgbat = rgb2Yuv(texIn[uvt]);
	float4 rgbab = rgb2Yuv(texIn[uvb]);
	uint2 uIndex = uint2(0, height * 2) + uint2(DTid.x, DTid.y);
	uint2 vIndex = uint2(0, height * 3) + uint2(DTid.x, DTid.y);
	texOut[uvt] = rgbat.x;
	texOut[uvb] = rgbab.x;
	texOut[uIndex] = (rgbat.y + rgbab.y) / 2.0f;
	texOut[vIndex] = (rgbat.z + rgbab.z) / 2.0f;
#endif
	//OEIP_YUVFMT_YUV420SP || OEIP_YUVFMT_YUV420P
#if (OEIP_YUV_TYPE == 1 || OEIP_YUV_TYPE == 6)
	uint2 uvlt = uint2(DTid.x * 2, DTid.y * 2);
	uint2 uvlb = uint2(DTid.x * 2, DTid.y * 2 + 1);
	uint2 uvrt = uint2(DTid.x * 2 + 1, DTid.y * 2);
	uint2 uvrb = uint2(DTid.x * 2 + 1, DTid.y * 2 + 1);
	float4 rgbalt = rgb2Yuv(texIn[uvlt]);
	float4 rgbalb = rgb2Yuv(texIn[uvlb]);
	float4 rgbart = rgb2Yuv(texIn[uvrt]);
	float4 rgbarb = rgb2Yuv(texIn[uvrb]);
#if (OEIP_YUV_TYPE == 1)
	uint2 uIndex = uint2(0, height * 2) + uint2(DTid.x * 2, DTid.y);
	uint2 vIndex = uint2(0, height * 2) + uint2(DTid.x * 2 + 1, DTid.y);
#elif (OEIP_YUV_TYPE == 6)
	uint2 nuv = u12u2(u22u1(DTid.xy, width), width * 2);
	uint2 uIndex = uint2(0, height * 2) + nuv;
	uint2 vIndex = uint2(0, height * 5 / 2) + nuv;
#endif
	texOut[uvlt] = rgbalt.x;
	texOut[uvlb] = rgbalb.x;
	texOut[uvrt] = rgbart.x;
	texOut[uvrb] = rgbarb.x;
	texOut[uIndex] = (rgbalt.y + rgbalt.y + rgbart.y + rgbarb.y) / 4.0f;
	texOut[vIndex] = (rgbalt.z + rgbalt.z + rgbart.z + rgbarb.z) / 4.0f;
#endif

	//OEIP_YUVFMT_YUV2 || OEIP_YUVFMT_YVYUI || OEIP_YUVFMT_UYVYI
#if (OEIP_YUV_TYPE == 2 || OEIP_YUV_TYPE == 3 || OEIP_YUV_TYPE == 4)
	//OEIP_YUVFMT_YUV2
	int bitx = 0;
	int yoffset = 0;
#if (OEIP_YUV_TYPE == 3)//OEIP_YUVFMT_YVYUI
	bitx = 2;
#elif (OEIP_YUV_TYPE == 4)//OEIP_YUVFMT_UYVYI
	yoffset = 1;
#endif
	float4 rgba1 = texIn[uint2(DTid.x * 2, DTid.y)];
	float4 rgba2 = texIn[uint2(DTid.x * 2 + 1, DTid.y)];
	float4 yuv1 = rgb2Yuv(rgba1);
	float4 yuv2 = rgb2Yuv(rgba2);
	float4 yuyv = float4(yuv1.x, (yuv1.y + yuv2.y) / 2.f, yuv2.x, (yuv1.z + yuv2.z) / 2.f);
	float4 syuyv = float4(yuyv[yoffset], yuyv[bitx + (1 - yoffset)], yuyv[yoffset + 2], yuyv[(2 - bitx) + (1 - yoffset)]);
	texOut[DTid.xy] = syuyv;
#endif
}