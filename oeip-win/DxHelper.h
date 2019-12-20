#pragma once
#ifdef OEIPWIN_EXPORT
#define OEIPWINDLL_EXPORT __declspec(dllexport)
#else
#define OEIPWINDLL_EXPORT __declspec(dllimport)
#endif

#include <d3dcommon.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <atlcomcli.h>
#include <cstdint>
#include "../oeip/OeipCommon.h"

OEIPWINDLL_EXPORT int sizeDxFormatElement(DXGI_FORMAT format);

OEIPWINDLL_EXPORT DXGI_FORMAT getDxFormat(int32_t dataType);

//OEIPWINDLL_EXPORT DXGI_FORMAT getDxFormat(OeipImageType oeipFormat);

