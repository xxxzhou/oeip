#pragma once
#include "LayerDx11.h"

struct BlendConstant
{
	InputConstant inputConstant = {};
	BlendParamet blendParamet = {};
};

class BlendLayerDx11 : public BlendLayer, public LayerDx11
{
public:
	BlendLayerDx11() :LayerDx11(2, 1) {
		computeShader = std::make_unique<Dx11ComputeShader>();
		computeShader->setCS(112, modeName, rctype);
	};
	~BlendLayerDx11() {};
private:
	BlendConstant blendConstant = {};
protected:
	virtual void onParametChange(BlendParamet oldT) override;
	virtual bool initHlsl() override;
	virtual bool onInitCBuffer() override;
	virtual bool updateCBuffer() override;
};

