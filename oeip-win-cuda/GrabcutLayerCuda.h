#pragma once

#include "KmeansCuda.h"
#include "GraphCuda.h"
#include "GMMCuda.h"
#include "LayerCuda.h"

class GrabcutLayerCuda : public GrabcutLayer, public LayerCuda
{
public:
	GrabcutLayerCuda();
	~GrabcutLayerCuda();
private:

protected:
	virtual void onParametChange(GrabcutParamet oldT) override;
	virtual bool onInitBuffer() override;
	virtual void onRunLayer() override;
};

