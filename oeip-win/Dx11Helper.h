#pragma once
#include "DxHelper.h"
//创建一个Dx11环境
OEIPWINDLL_EXPORT bool createDevice11(ID3D11Device** deviceDx11, ID3D11DeviceContext** ctxDx11);
//从CPU数据传入GPU中
OEIPWINDLL_EXPORT bool updateDx11Resource(ID3D11DeviceContext* ctxDx11, ID3D11Resource* resouce, uint8_t* data, uint32_t size);
//byteWidth不一定和width*elementSize相等，需要注意处理
OEIPWINDLL_EXPORT bool downloadDx11Resource(ID3D11DeviceContext* ctxDx11, ID3D11Resource* resouce, uint8_t** data, uint32_t& byteWidth);
//DX11截屏
OEIPWINDLL_EXPORT bool createGUITextureBuffer(ID3D11Device* deviceDx11, int width, int height, ID3D11Texture2D** ppBufOut);
//用于创建一块CPU不能读的BUFFER到相同规格CPU能读的BUFFER
OEIPWINDLL_EXPORT void copyBufferToRead(ID3D11Device* deviceDx11, ID3D11Buffer* pBuffer, ID3D11Buffer** descBuffer);
OEIPWINDLL_EXPORT void copyBufferToRead(ID3D11Device* deviceDx11, ID3D11Texture2D* pBuffer, ID3D11Texture2D** descBuffer);
//SRV：描述了诸如贴图等，由Shader进行只读访问的内存空间，一般用来做CS中的输入
//UAV：描述了可以由Shader进行随机读写的内存空间，用于做CS中的输入与输出
OEIPWINDLL_EXPORT bool createBufferSRV(ID3D11Device* deviceDx11, ID3D11Buffer* pBuffer, ID3D11ShaderResourceView** ppSRVOut);
OEIPWINDLL_EXPORT bool createBufferUAV(ID3D11Device* deviceDx11, ID3D11Buffer* pBuffer, ID3D11UnorderedAccessView** ppUAVOut);
OEIPWINDLL_EXPORT bool createBufferSRV(ID3D11Device* deviceDx11, ID3D11Texture2D* pBuffer, ID3D11ShaderResourceView** ppSRVOut);
OEIPWINDLL_EXPORT bool createBufferUAV(ID3D11Device* deviceDx11, ID3D11Texture2D* pBuffer, ID3D11UnorderedAccessView** ppUAVOut);

OEIPWINDLL_EXPORT HANDLE getDx11SharedHandle(ID3D11Resource* source);

OEIPWINDLL_EXPORT void copySharedToTexture(ID3D11Device* d3ddevice, HANDLE& sharedHandle, ID3D11Texture2D* texture);

OEIPWINDLL_EXPORT void copyTextureToShared(ID3D11Device* d3ddevice, HANDLE& sharedHandle, ID3D11Texture2D* texture);



