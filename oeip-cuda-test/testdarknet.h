#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "../oeip/OeipExport.h"
#include "../oeip/BaseLayer.h"
#include "../oeip/VideoPipe.h"

namespace DarknetPerson
{
	VideoPipe* vpipe = nullptr;
	void testDarknetPerson() {
		initOeip();

		vpipe = new VideoPipe(OEIP_CUDA);
		outMap = addPiepLayer(vpipe->getPipeId(), "darknet", OEIP_DARKNET_LAYER);

	}
}