
#ifndef SIZE_X
#define SIZE_X 32
#endif

#ifndef SIZE_Y
#define SIZE_Y 8
#endif

#define UINT82FLOAT 0.00392156862745f

uint u22u1(uint2 uv, uint step) {
	return uv.y * step + uv.x;
}

uint2 u12u2(uint index, uint step) {
	uint2 uv = 0;
	uv.x = index % step;
	uv.y = index / step;
	return uv;
}

float4 yuv2Rgb(float y, float u, float v, float a) {
	float4 xrgba = 0.f;
	xrgba.r = clamp(y + 1.402f * v, 0.f, 1.f);
	xrgba.g = clamp(y - 0.71414f * v - 0.34414f * u, 0.f, 1.f);
	xrgba.b = clamp(y + 1.772f * u, 0.f, 1.f);
	xrgba.a = a;
	return xrgba;
}

float4 uchar2float(uint4 rgba) {
	float4 xrgba = float4(rgba * UINT82FLOAT);
	//xrgba.r = clamp(rgba.r * UINT82FLOAT, 0.f, 1.f);
	//xrgba.g = clamp(rgba.g * UINT82FLOAT, 0.f, 1.f);
	//xrgba.b = clamp(rgba.b * UINT82FLOAT, 0.f, 1.f);
	//xrgba.a = clamp(rgba.a * UINT82FLOAT, 0.f, 1.f);
	return xrgba;
}

uint4 uint32Unzip(uint data) {
	uint4 xrgba = uint4(data & 0x000000FF, (data & 0x0000FF00) >> 8, (data & 0x00FF0000) >> 16, (data & 0xFF000000) >> 24);
	return xrgba;
}

uint uint8Zip(uint4 rgba) {
	uint data = rgba.x | rgba.y << 8 | rgba.z << 16 | rgba.w << 24;
	return data;
}

float4 rgb2Yuv(float4 rgba) {
	float4 yuva;
	//xyz -> yuv
	yuva.x = clamp(0.299 * rgba.r + 0.587 * rgba.g + 0.114 * rgba.b, 0, 1);
	//uv (-0.5,0.5)
	yuva.y = clamp(-0.1687 * rgba.r - 0.3313 * rgba.g + 0.5 * rgba.b + 0.5f, 0, 1);
	yuva.z = clamp(0.5 * rgba.r - 0.4187 * rgba.g - 0.0813 * rgba.b + 0.5f, 0, 1);
	yuva.a = clamp(rgba.w, 0, 1);
	return yuva;
}

//Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16
//U = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128
//V = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128

//Y = round(0.256788 * R + 0.504129 * G + 0.097906 * B) + 16
//U = round(-0.148223 * R - 0.290993 * G + 0.439216 * B) + 128
//V = round(0.439216 * R - 0.367788 * G - 0.071427 * B) + 128