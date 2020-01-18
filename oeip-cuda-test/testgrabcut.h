#pragma once

#include "../oeip-win-cuda/GrabcutCuda.h"

void rgb2rgba_gpu(PtrStepSz<uchar3> source, PtrStepSz<uchar4> dest, cudaStream_t stream = nullptr);
void setMask_gpu(PtrStepSz<uchar> mask, int x, int y, int radius, int vmask, cudaStream_t stream = nullptr);
void showMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);
void showSeedMask_gpu(PtrStepSz<uchar4> source, PtrStepSz<uchar> mask, cudaStream_t stream = nullptr);


namespace Grabcut
{
	float gamma = 90.f;
	float lambda = 250.f;
	int maxCount = 250;
	int iterCount = 1;
	int bCpu = 0;
	int bSeed = 0;
	bool bComputeSeed = false;
	int vMask = 0;
	bool bMove = 0;
	int xx = 0;
	int yy = 0;
	int radius = 3;
	bool bReset = false;
	bool bCompute = false;
	int personCount = 0;

	static void on_trackbar(int, void*) {
	}

	void createUI(const char *uiName) {
		int igamma = gamma;
		int ilambda = lambda;

		cv::namedWindow(uiName);
		cv::resizeWindow(uiName, 400, 400);
		cv::createTrackbar("iterCount", uiName, &iterCount, 10, on_trackbar);
		cv::createTrackbar("gamma", uiName, &igamma, 300, on_trackbar);
		cv::createTrackbar("lambda", uiName, &ilambda, 500, on_trackbar);
		cv::createTrackbar("maxCount", uiName, &maxCount, 2000, on_trackbar);
		cv::createTrackbar("bCpu", uiName, &bCpu, 1, on_trackbar);
		cv::createTrackbar("bSeed", uiName, &bSeed, 1, on_trackbar);
		cv::createTrackbar("radius", uiName, &radius, 10, on_trackbar);
	}

	void updateUI(const char *uiName) {
		iterCount = cv::getTrackbarPos("iterCount", uiName);
		gamma = cv::getTrackbarPos("gamma", uiName) + 1;
		lambda = cv::getTrackbarPos("lambda", uiName) + 1;
		maxCount = cv::getTrackbarPos("maxCount", uiName);
		bCpu = cv::getTrackbarPos("bCpu", uiName);
		bSeed = cv::getTrackbarPos("bSeed", uiName);
		radius = cv::getTrackbarPos("radius", uiName);
	}

	void on_MouseHandle(int event, int x, int y, int flags, void* param) {
		if (event == cv::EVENT_LBUTTONDOWN)
			bMove = true;
		if (event == cv::EVENT_LBUTTONUP)
			bMove = false;
		switch (event) {
		case cv::EVENT_MOUSEMOVE: {
			xx = x;
			yy = y;
			break;
		}
		default:
			break;
		};
	}

	void testGrabcut() {
		cv::VideoCapture cap;
		cv::Mat frame;
		cv::Mat result, reframe;

		cap.open(1);

		cv::namedWindow("image");
		cv::namedWindow("seg image");

		int height = 416;
		int width = 416;
		GrabcutCuda grabCut = {};
		grabCut.init(width, height);
		GpuMat gpuFrame;
		createUI("1 image");
		GpuMat x4source = GpuMat(height, width, CV_8UC4);
		cv::Mat bgModel, fgModel;
		cv::Mat foreground(cv::Size(width, height), CV_8UC3, cv::Scalar(0, 0, 0));
		while (int key = cv::waitKey(10)) {
			//thread ex = thread(updateUI, "1 image");
			//ex.detach();
			updateUI("1 image");

			cap >> frame;
			cv::resize(frame, reframe, cv::Size(width, height));
			cv::Rect rectangle(120, 26, 150, 207);
			if (bCpu) {
				cv::grabCut(reframe, result, rectangle, bgModel, fgModel, iterCount, cv::GC_INIT_WITH_RECT);
			}
			else {
				int a = 0;
				int c = {};
				int b;
				uint d;
				grabCut.setParams(iterCount, gamma, lambda, maxCount);
				cudaDeviceSynchronize();
				gpuFrame.upload(reframe);
				rgb2rgba_gpu(gpuFrame, x4source);
				if (bSeed)
					grabCut.renderFrame(x4source);
				else
					grabCut.renderFrame(x4source, rectangle);
				grabCut.mask.download(result);
			}

			cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
			//cv::Mat foreground(reframe.size(), CV_8UC3, cv::Scalar(0, 0, 0));
			foreground = cv::Scalar(0, 0, 0);
			reframe.copyTo(foreground, result);
			cv::imshow("image", foreground);
			cv::rectangle(reframe, rectangle, cv::Scalar(0, 0, 255), 3);
			cv::imshow("seg image", reframe);
		}
	}

	//种子点计算模式
	void testSeedGrabCude() {
		cv::VideoCapture cap;
		cv::Mat frame;
		cv::Mat result, reframe;
		cap.open(1);

		cv::namedWindow("image");
		cv::namedWindow("seg image");

		GrabcutCuda grabCut;
		int height = 416;
		int width = 416;

		//lambda = 20.f;
		maxCount = 50;
		grabCut.init(width, height);
		GpuMat gpuFrame;
		createUI("1 image");
		cudaMemset2D(grabCut.mask.ptr(), grabCut.mask.step, 3, grabCut.mask.cols, grabCut.mask.rows);
		GpuMat x4source = GpuMat(height, width, CV_8UC4);
		cv::setMouseCallback("seg image", on_MouseHandle, nullptr);
		cv::Mat foreground(reframe.size(), CV_8UC3, cv::Scalar(0, 0, 0));

		while (int key = cv::waitKey(2)) {
			updateUI("1 image");
			if (key == 'q') {
				bComputeSeed = !bComputeSeed;
			}
			else if (key == 'w') {
				vMask = 1;
			}
			else if (key == 'e') {
				vMask = 0;
			}
			else if (key == 'r') {
				bCompute = true;
			}
			else if (key == 'a') {
				bReset = true;
			}
			cap >> frame;
			cv::resize(frame, reframe, cv::Size(width, height));

			grabCut.setParams(iterCount, gamma, lambda, maxCount);
			cudaDeviceSynchronize();
			gpuFrame.upload(reframe);
			rgb2rgba_gpu(gpuFrame, x4source);
			if (bComputeSeed) {
				if (bReset) {
					cudaMemset2D(grabCut.mask.ptr(), grabCut.mask.step, 3, grabCut.mask.cols, grabCut.mask.rows);
					bReset = false;
				}
				if (bMove) {
					setMask_gpu(grabCut.mask, xx, yy, radius, vMask);
				}
				if (bCompute) {
					grabCut.computeSeed(x4source);
					bCompute = false;
					bComputeSeed = false;
				}
				showSeedMask_gpu(x4source, grabCut.mask);
				cv::Mat foreground;
				x4source.download(foreground);
				cv::imshow("image", foreground);
				cv::imshow("seg image", reframe);
			}
			else {
				cudaMemset2D(grabCut.mask.ptr(), grabCut.mask.step, 3, grabCut.mask.cols, grabCut.mask.rows);
				grabCut.renderFrame(x4source);
				showMask_gpu(x4source, grabCut.mask);
				cv::Mat foreground1;
				x4source.download(foreground1);
				cv::imshow("image", foreground1);
				cv::imshow("seg image", reframe);
			}
		}
	}

}
