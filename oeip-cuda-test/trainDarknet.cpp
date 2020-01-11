#include "trainDarknet.h"
#include "../darknet/darknet.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

float constrain(float min, float max, float a) {
	if (a < min) return min;
	if (a > max) return max;
	return a;
}

void correct_boxes(box_label *boxes, int n, int flip, int vflip, int trans) {
	int i;
	for (i = 0; i < n; ++i) {
		if (boxes[i].x == 0 && boxes[i].y == 0) {
			boxes[i].x = 999999;
			boxes[i].y = 999999;
			boxes[i].w = 999999;
			boxes[i].h = 999999;
			continue;
		}
		boxes[i].left = boxes[i].left;
		boxes[i].right = boxes[i].right;
		boxes[i].top = boxes[i].top;
		boxes[i].bottom = boxes[i].bottom;

		if (flip)
		{
			float swap = boxes[i].left;
			boxes[i].left = 1. - boxes[i].right;
			boxes[i].right = 1. - swap;
		}
		if (vflip)
		{
			float swap = boxes[i].top;
			boxes[i].top = 1. - boxes[i].bottom;
			boxes[i].bottom = 1. - swap;
		}

		boxes[i].left = constrain(0, 1, boxes[i].left);
		boxes[i].right = constrain(0, 1, boxes[i].right);
		boxes[i].top = constrain(0, 1, boxes[i].top);
		boxes[i].bottom = constrain(0, 1, boxes[i].bottom);

		boxes[i].x = (boxes[i].left + boxes[i].right) / 2;
		boxes[i].y = (boxes[i].top + boxes[i].bottom) / 2;
		boxes[i].w = (boxes[i].right - boxes[i].left);
		boxes[i].h = (boxes[i].bottom - boxes[i].top);

		boxes[i].w = constrain(0, 1, boxes[i].w);
		boxes[i].h = constrain(0, 1, boxes[i].h);

		if (trans)
		{
			float temp = boxes[i].x;
			boxes[i].x = boxes[i].y;
			boxes[i].y = temp;
			temp = boxes[i].w;
			boxes[i].w = boxes[i].h;
			boxes[i].h = temp;
		}
	}
}

void transpose_image(image im) {
	if (im.h == im.w) {
		int n, m;
		int c;
		for (c = 0; c < im.c; ++c) {
			for (n = 0; n < im.w - 1; ++n) {
				for (m = n + 1; m < im.w; ++m) {
					float swap = im.data[m + im.w*(n + im.h*c)];
					im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
					im.data[n + im.w*(m + im.h*c)] = swap;
				}
			}
		}
	}
}

cv::Mat image_to_mat(image im, int flip, int vflip, int trans) {
	image copy = copy_image(im);
	if (flip)
		flip_image(copy);
	if (vflip)
		vflip_image(copy);
	if (trans)
		transpose_image(copy);
	constrain_image(copy);
	if (im.c == 3)
		rgbgr_image(copy);
	int type = im.c == 3 ? CV_8UC3 : CV_8UC1;
	cv::Mat m(copy.h, copy.w, type);
	int x, y, c;
	for (y = 0; y < im.h; ++y) {
		for (x = 0; x < im.w; ++x) {
			for (c = 0; c < im.c; ++c) {
				float val = copy.data[c*im.h*im.w + y * im.w + x];
				m.data[y*m.step + x * im.c + c] = (unsigned char)(val * 255);
			}
		}
	}
	return m;
}

void trainDarknet() {
	char *train_images = "../../ThirdParty/train.txt";
	char *cfgfile = "../../ThirdParty/yolov3-tiny.cfg";//yolov3-tiny.cfg
	char *backup_directory = "../../ThirdParty/saveweight";
	char *weightfile = "../../ThirdParty/saveweight/yolov3-tiny.backup";
	float avg_loss = -1;
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	network *net = load_network(cfgfile, weightfile, 0);
	std::cout << "learning rate:" << net->learning_rate << " momentum:" << net->momentum << " decay:" << net->decay << std::endl;

	int imgs = net->batch*net->subdivisions;
	int i = *net->seen / imgs;

	data train, buffer;

	layer l = net->layers[net->n - 1];

	int side = l.side;
	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
	char **paths = (char **)list_to_array(plist);

	load_args args = { 0 };
	args.w = net->w;
	args.h = net->h;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	//0.3
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = REGION_DATA;

	args.angle = net->angle;
	args.exposure = net->exposure;
	args.saturation = net->saturation;
	args.hue = net->hue;

	clock_t time;
	while (get_current_batch(net) < net->max_batches) {
		i += 1;
		time = clock();
		//加载的数据会保存在args.d中
		train = load_data_detection(args.n, args.paths, args.m, args.w, args.h, args.num_boxes,
			args.classes, args.jitter, args.hue, args.saturation, args.exposure);
		printf("Loaded: %lf seconds\n", sec(clock() - time));
		time = clock();
		float loss = train_network(net, train);
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss * .9 + loss * .1;

		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock() - time), i*imgs);
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		if (i % 100 == 0) {
			char buff[256];
			sprintf(buff, "%s/%s.backup", backup_directory, base);
			save_weights(net, buff);
		}
		free_data(train);
	}
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

void validateDarknet() {

	//cv::Mat frame;
	//cv::Mat sframe;
	cv::namedWindow("image");
	cv::namedWindow("simage");

	char *train_images = "../../ThirdParty/test.txt";
	//char *cfgfile = "D:/WorkSpace/DeepLearning/cocodata/yolov3.cfg";//yolov3-tiny.cfg	
	//char *weightfile = "D:/WorkSpace/DeepLearning/cocodata/yolov3.weights";
	char *cfgfile = "../../ThirdParty/yolov3-tiny-test.cfg";//yolov3-tiny.cfg/yolov3.cfg	
	char *weightfile = "../../ThirdParty/yolov3-tiny_745000.weights";//yolov3.weights
	float avg_loss = -1;
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	network *net = load_network(cfgfile, weightfile, 0);
	std::cout << "learning rate:" << net->learning_rate << " momentum:" << net->momentum << " decay:" << net->decay << std::endl;

	list *plist = get_paths(train_images);
	char **paths = (char **)list_to_array(plist);

	int width = net->w;
	int height = net->h;

	layer l = net->layers[net->n - 1];

	int j, k;

	int m = plist->size;
	float thresh = .22;
	float iou_thresh = .5;
	float nms = .4;
	int i = 0;
	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;
	//for (i = 0; i < m; ++i)
	while (true) {
		int flip = rand() % 2;
		int vflip = rand() % 2;
		int trans = rand() % 2;

		//i++;
		i = rand() % m;
		char *path = paths[i];
		image orig = load_image_color(path, 0, 0);
		image sized = resize_image(orig, net->w, net->h);
		if (flip)
			flip_image(sized);
		if (vflip)
			vflip_image(sized);
		if (trans)
			transpose_image(sized);
		char *id = basecfg(path);
		network_predict(net, sized.data);
		int nboxes = 0;
		//thresh越小会引进更多box,objectness比thresh大的都引进
		detection *dets = get_network_boxes(net, sized.w, sized.h, thresh, thresh, 0, 1, &nboxes);
		//nms越小会去掉更多的box,nms越小，相近的box都会去掉
		if (nms)
			do_nms_obj(dets, nboxes, 1, nms);
		for (k = 0; k < nboxes; ++k) {
			if (dets[k].objectness > thresh) {
				++proposals;
			}
		}
		char labelpath[4096];
		find_replace(path, "yolotestdest", "yolotestlabel", labelpath);
		find_replace(labelpath, ".jpg", ".txt", labelpath);
		find_replace(labelpath, ".png", ".txt", labelpath);
		find_replace(labelpath, ".JPG", ".txt", labelpath);
		find_replace(labelpath, ".JPEG", ".txt", labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);

		//for (j = 0; j < num_labels; ++j) {
		//	++total;
		//	box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
		//	float best_iou = 0;
		//	for (k = 0; k < l.w*l.h*l.n; ++k) {
		//		float iou = box_iou(dets[k].bbox, t);
		//		if (dets[k].objectness > thresh && iou > best_iou) {
		//			best_iou = iou;
		//		}
		//	}
		//	avg_iou += best_iou;
		//	if (best_iou > iou_thresh) {
		//		++correct;
		//	}
		//}
		//int flip = rand() % 2;
		//int vflip = rand() % 2;
		//int trans = rand() % 2;
		cv::Mat mat = image_to_mat(sized, 0, 0, 0);
		correct_boxes(truth, num_labels, flip, vflip, trans);
		cv::Mat sframe = image_to_mat(orig, 0, 0, 0);
		cv::Rect rectangle;
		for (int i = 0; i < nboxes; ++i) {
			if (dets[i].objectness > thresh) {
				rectangle.width = dets[i].bbox.w*width;
				rectangle.height = dets[i].bbox.h*height;
				rectangle.x = dets[i].bbox.x*width - rectangle.width / 2;
				rectangle.y = dets[i].bbox.y*height - rectangle.height / 2;
				cv::putText(mat, std::to_string(dets[i].objectness), cv::Point(rectangle.x, rectangle.y), 1, 2, cv::Scalar(0, 0, 255));
				cv::rectangle(mat, rectangle, cv::Scalar(0, 0, 255), 1);
			}
		}
		for (int i = 0; i < num_labels; i++) {
			rectangle.width = truth[i].w*width;
			rectangle.height = truth[i].h*height;
			rectangle.x = truth[i].x * width - rectangle.width / 2;
			rectangle.y = truth[i].y * height - rectangle.height / 2;
			cv::rectangle(mat, rectangle, cv::Scalar(0, 255, 255), 1);
		}

		cv::imshow("image", mat);
		cv::imshow("simage", sframe);
		fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
		free(id);
		free_image(orig);
		free_image(sized);

		int key = cv::waitKey();
		if (key == 'n') {
			continue;
		}
	}
}