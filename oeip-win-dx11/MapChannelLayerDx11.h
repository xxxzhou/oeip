#pragma once
#include "LayerDx11.h"

struct MapChannelConstant
{
	InputConstant inputConstant = {};
	MapChannelParamet mapChannelParamet = {};
};

class MapChannelLayerDx11 : public MapChannelLayer, public LayerDx11
{
public:
	MapChannelLayerDx11();
	~MapChannelLayerDx11();
private:
	MapChannelConstant mapConstant = {};
protected:
	virtual void onParametChange(MapChannelParamet oldT) override;
	virtual bool initHlsl() override;
	virtual bool onInitCBuffer() override;
	virtual bool updateCBuffer() override;
};

