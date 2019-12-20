#pragma once

//采用opencv里的数据类型,前三BIT表示类型，后三BIT表示通道个数
#pragma region opencv 
#define OEIP_CV_CN_MAX     512
#define OEIP_CV_CN_SHIFT   3
#define OEIP_CV_DEPTH_MAX  (1 << OEIP_CV_CN_SHIFT)

#define OEIP_CV_8U   0
#define OEIP_CV_8S   1
#define OEIP_CV_16U  2
#define OEIP_CV_16S  3
#define OEIP_CV_32S  4
#define OEIP_CV_32F  5
#define OEIP_CV_64F  6
#define OEIP_CV_16F  7

#define OEIP_CV_MAT_DEPTH_MASK       (OEIP_CV_DEPTH_MAX - 1)
#define OEIP_CV_MAT_DEPTH(flags)     ((flags) & OEIP_CV_MAT_DEPTH_MASK)

#define OEIP_CV_MAKETYPE(depth,cn) (OEIP_CV_MAT_DEPTH(depth) + (((cn)-1) << OEIP_CV_CN_SHIFT))
#define OEIP_CV_MAKE_TYPE OEIP_CV_MAKETYPE

#define OEIP_CV_8UC1 OEIP_CV_MAKETYPE(OEIP_CV_8U,1)
#define OEIP_CV_8UC2 OEIP_CV_MAKETYPE(OEIP_CV_8U,2)
#define OEIP_CV_8UC3 OEIP_CV_MAKETYPE(OEIP_CV_8U,3)
#define OEIP_CV_8UC4 OEIP_CV_MAKETYPE(OEIP_CV_8U,4)
#define OEIP_CV_8UC(n) OEIP_CV_MAKETYPE(OEIP_CV_8U,(n))

#define OEIP_CV_8SC1 OEIP_CV_MAKETYPE(OEIP_CV_8S,1)
#define OEIP_CV_8SC2 OEIP_CV_MAKETYPE(OEIP_CV_8S,2)
#define OEIP_CV_8SC3 OEIP_CV_MAKETYPE(OEIP_CV_8S,3)
#define OEIP_CV_8SC4 OEIP_CV_MAKETYPE(OEIP_CV_8S,4)
#define OEIP_CV_8SC(n) OEIP_CV_MAKETYPE(OEIP_CV_8S,(n))

#define OEIP_CV_16UC1 OEIP_CV_MAKETYPE(OEIP_CV_16U,1)
#define OEIP_CV_16UC2 OEIP_CV_MAKETYPE(OEIP_CV_16U,2)
#define OEIP_CV_16UC3 OEIP_CV_MAKETYPE(OEIP_CV_16U,3)
#define OEIP_CV_16UC4 OEIP_CV_MAKETYPE(OEIP_CV_16U,4)
#define OEIP_CV_16UC(n) OEIP_CV_MAKETYPE(OEIP_CV_16U,(n))

#define OEIP_CV_16SC1 OEIP_CV_MAKETYPE(OEIP_CV_16S,1)
#define OEIP_CV_16SC2 OEIP_CV_MAKETYPE(OEIP_CV_16S,2)
#define OEIP_CV_16SC3 OEIP_CV_MAKETYPE(OEIP_CV_16S,3)
#define OEIP_CV_16SC4 OEIP_CV_MAKETYPE(OEIP_CV_16S,4)
#define OEIP_CV_16SC(n) OEIP_CV_MAKETYPE(OEIP_CV_16S,(n))

#define OEIP_CV_32SC1 OEIP_CV_MAKETYPE(OEIP_CV_32S,1)
#define OEIP_CV_32SC2 OEIP_CV_MAKETYPE(OEIP_CV_32S,2)
#define OEIP_CV_32SC3 OEIP_CV_MAKETYPE(OEIP_CV_32S,3)
#define OEIP_CV_32SC4 OEIP_CV_MAKETYPE(OEIP_CV_32S,4)
#define OEIP_CV_32SC(n) OEIP_CV_MAKETYPE(OEIP_CV_32S,(n))

#define OEIP_CV_32FC1 OEIP_CV_MAKETYPE(OEIP_CV_32F,1)
#define OEIP_CV_32FC2 OEIP_CV_MAKETYPE(OEIP_CV_32F,2)
#define OEIP_CV_32FC3 OEIP_CV_MAKETYPE(OEIP_CV_32F,3)
#define OEIP_CV_32FC4 OEIP_CV_MAKETYPE(OEIP_CV_32F,4)
#define OEIP_CV_32FC(n) OEIP_CV_MAKETYPE(OEIP_CV_32F,(n))

#define OEIP_CV_64FC1 OEIP_CV_MAKETYPE(OEIP_CV_64F,1)
#define OEIP_CV_64FC2 OEIP_CV_MAKETYPE(OEIP_CV_64F,2)
#define OEIP_CV_64FC3 OEIP_CV_MAKETYPE(OEIP_CV_64F,3)
#define OEIP_CV_64FC4 OEIP_CV_MAKETYPE(OEIP_CV_64F,4)
#define OEIP_CV_64FC(n) OEIP_CV_MAKETYPE(OEIP_CV_64F,(n))

#define OEIP_CV_16FC1 OEIP_CV_MAKETYPE(OEIP_CV_16F,1)
#define OEIP_CV_16FC2 OEIP_CV_MAKETYPE(OEIP_CV_16F,2)
#define OEIP_CV_16FC3 OEIP_CV_MAKETYPE(OEIP_CV_16F,3)
#define OEIP_CV_16FC4 OEIP_CV_MAKETYPE(OEIP_CV_16F,4)
#define OEIP_CV_16FC(n) OEIP_CV_MAKETYPE(OEIP_CV_16F,(n))

#define OEIP_CV_MAT_CN_MASK          ((OEIP_CV_CN_MAX - 1) << OEIP_CV_CN_SHIFT)
#define OEIP_CV_MAT_CN(flags)        ((((flags) & OEIP_CV_MAT_CN_MASK) >> OEIP_CV_CN_SHIFT) + 1)
/** Size of each channel item,
   0x28442211 = 0010 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define OEIP_CV_ELEM_SIZE1(type) ((0x28442211 >> OEIP_CV_MAT_DEPTH(type)*4) & 15)

#define OEIP_CV_ELEM_SIZE(type) (OEIP_CV_MAT_CN(type)*OEIP_CV_ELEM_SIZE1(type))
#pragma endregion