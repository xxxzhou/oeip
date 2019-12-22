#pragma once
#include "LayerDx11.h"
class RGBA2YUVLayerDX11 : public RGBA2YUVLayer, public LayerDx11
{
public:
	RGBA2YUVLayerDX11();
	~RGBA2YUVLayerDX11() {};
private:
	void initConnect();
protected:
	virtual void onParametChange(RGBA2YUVParamet oldT) override;
	virtual bool initHlsl() override;
	virtual void onInitLayer() override;
};

