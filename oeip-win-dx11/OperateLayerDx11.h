#pragma once
#include "LayerDx11.h"
struct OperateConstant
{
	InputConstant inputConstant = {};
	OperateParamet operateParamet = {};
};

class OperateLayerDx11 : public OperateLayer, public LayerDx11
{
public:
	OperateLayerDx11();
	~OperateLayerDx11();
private:
	OperateConstant operateConstant = {};
protected:
	virtual void onParametChange(OperateParamet oldT) override;
	virtual bool initHlsl() override;
	virtual bool onInitCBuffer() override;
	virtual bool updateCBuffer() override;
};

