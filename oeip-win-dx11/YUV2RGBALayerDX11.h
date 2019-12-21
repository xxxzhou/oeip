#pragma once
#include "LayerDx11.h"

class YUV2RGBALayerDX11 : public YUV2RGBALayer, public LayerDx11
{
public:
	YUV2RGBALayerDX11();
	~YUV2RGBALayerDX11() {};
private:
	void initConnect();
protected:
	virtual void onParametChange(YUV2RGBAParamet oldT) override;
	virtual bool initHlsl() override;
	virtual void onInitLayer() override;
};

