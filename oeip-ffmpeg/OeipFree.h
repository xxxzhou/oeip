#pragma once
#include "Oeipffmpeg.h"

//std::unique_ptr<HDC__, std::function<void(HDC)>>
	//	target_hdc(GetWindowDC(hwnd), [=](HDC x) { ReleaseDC(hwnd, x); });

template<typename T>
void freefobj(T* val) {
	av_free(val);
}

template<>
inline void freefobj(AVCodecContext* val) {
	avcodec_close(val);
	avcodec_free_context(&val);
}

template<>
inline void freefobj(AVFormatContext* val) {
/*	if (val->flags & AVFMT_NOFILE) {
		avio_close(val->pb);
		avformat_free_context(val);
	}*/	
	avformat_close_input(&val);
}

template<>
inline void freefobj(AVFrame* val) {
	av_frame_free(&val);
}

template<>
inline void freefobj(SwrContext* val) {
	swr_free(&val);
}

template<>
inline void freefobj(AVIOContext* val) {
	av_freep(&val->buffer);
	av_free(val);
}

#define OEIP_UNIQUE_FUNCTION(CLASSTYPE) \
	typedef void (*free##CLASSTYPE) (CLASSTYPE*);

#define OEIP_UNIQUE_FFCLASS(CLASSTYPE)\
	OEIP_UNIQUE_FUNCTION(CLASSTYPE) \
	typedef std::unique_ptr<CLASSTYPE, free##CLASSTYPE> FO##CLASSTYPE;

//使用decltype，相应类型做为类的字段会声明不了
//typedef std::unique_ptr<CLASSTYPE, decltype(freefobj<CLASSTYPE>)*> O##CLASSTYPE;
#define OEIP_UNIQUE_FCLASS(CLASSTYPE) \
	typedef std::unique_ptr<CLASSTYPE, std::function<void(CLASSTYPE*)>> O##CLASSTYPE; \
	inline O##CLASSTYPE getUniquePtr(CLASSTYPE* ptr){ \
		O##CLASSTYPE uptr(ptr,freefobj<CLASSTYPE>);\
		return uptr;\
	}

OEIP_UNIQUE_FCLASS(AVFormatContext)
OEIP_UNIQUE_FCLASS(AVCodecContext)
OEIP_UNIQUE_FCLASS(AVFrame)
OEIP_UNIQUE_FCLASS(SwrContext)
OEIP_UNIQUE_FCLASS(AVIOContext)


