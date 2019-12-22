#pragma once
#include "LayerDx11.h"

class ResizeLayerDx11 : public ResizeLayer,public LayerDx11
{
public:
	ResizeLayerDx11();
	~ResizeLayerDx11();
protected:
	virtual void onParametChange(ResizeParamet oldT) override;
	virtual bool initHlsl() override;
	virtual void onInitLayer() override;
};

