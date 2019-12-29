#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

//在yolov3中,n=3 total=9（mask在yolov3中分三组，分别是[0,1,2/3,4,5/6,7,8],n表示分组里几个数据,n*3=total)
//608*608下,第一组是19*19，第二组是38*38，第三组是76*76,每组检查对应索引里的anchors
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
	//在这假设在主支中,其中缩小5次,608/(2^5)=19,这个分支中,w=h=19
	int i;
	layer l = { 0 };
	l.type = YOLO;
	//检测几种类型边框(这分支对应上面anchors[6,7,8]这个用来检测大边框)
	l.n = n;
	//如上,在yolov3中，有大中小分别有三个边框聚合,一共是3*3=9
	//而在yolov3-tiny中,有大小分别三个边框聚合，一共是3*2=6
	l.total = total;
	//一般来说，训练为32,预测为1
	l.batch = batch;
	//主支,608/(2^5)=19
	l.h = h;
	l.w = w;
	//如上在主支中,每张特征图有19*19个元素,c表示特征图个数,n表示对应的anchors[6,7,8]这三个
	//4表示box坐标,1是Po(预测机率与IOU正确率)的概率,classes是预测的类别数
	l.c = n * (classes + 4 + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	//检测一共有多少个类别
	l.classes = classes;
	//计算代价(数据集整体的误差描述)
	l.cost = calloc(1, sizeof(float));
	//对应表示anchors
	l.biases = calloc(total * 2, sizeof(float));
	//对应如上anchors中n对应需要用到的索引
	if (mask) l.mask = mask;
	else {
		l.mask = calloc(n, sizeof(int));
		for (i = 0; i < n; ++i) {
			l.mask[i] = i;
		}
	}
	l.bias_updates = calloc(n * 2, sizeof(float));
	//当前层batch为1的所有输出特征图包含的元素个数，每个元素为一个float
	l.outputs = h * w*n*(classes + 4 + 1);
	//当前层batch为1时所有输入特征图包含的元素个数，每个元素为一个float
	l.inputs = l.outputs;
	//标签(真实)数据,这里90表示如上w*h(19*19中)每个格子最多有90个label。
	//而每个label前面4个float表示box的四个点,后面1个float表示当前类别
	l.truths = 90 * (4 + 1);
	//计算误差(数据单个的误差描述),用来表示 期望输出-真实输出
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));
	for (i = 0; i < total * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_yolo_layer;
	l.backward = backward_yolo_layer;
#if GPU
	l.forward_gpu = forward_yolo_layer_gpu;
	l.backward_gpu = backward_yolo_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "yolo\n");
	srand(0);

	return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->outputs = h * w*l->n*(l->classes + 4 + 1);
	l->inputs = l->outputs;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));

#if GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}
//这里的n表示对应分组mask里的索引
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	box b;
	b.x = (i + x[index + 0 * stride]) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
	box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
	float iou = box_iou(pred, truth);
	//如在19*19中,tx,ty指的是19*19中的索引
	float tx = (truth.x*lw - i);
	float ty = (truth.y*lh - j);
	//y=log(x) y(0-正无穷) 
	float tw = log(truth.w*w / biases[2 * n]);
	float th = log(truth.h*h / biases[2 * n + 1]);

	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
	return iou;
}

//object confidence, class score 的delta class真实类别索引
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
	int n;
	if (delta[index]) {
		delta[index + stride * class] = 1 - output[index + stride * class];
		if (avg_cat) *avg_cat += output[index + stride * class];
		return;
	}
	for (n = 0; n < classes; ++n) {
		delta[index + stride * n] = ((n == class) ? 1 : 0) - output[index + stride * n];
		if (n == class && avg_cat)
			*avg_cat += output[index + stride * n];
	}
}

static int entry_index(layer l, int batch, int location, int entry)
{
	int n = location / (l.w*l.h);
	int loc = location % (l.w*l.h);
	return batch * l.outputs + n * l.w*l.h*(4 + l.classes + 1) + entry * l.w*l.h + loc;
}

void forward_yolo_layer(const layer l, network net)
{
	int i, j, b, t, n;
	memcpy(l.output, net.input, l.outputs*l.batch * sizeof(float));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array(l.output + index, 2 * l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array(l.output + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
		}
	}
#endif

	memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
	if (!net.train)
		return;
	//表示当前层正确配对的box的交并比的平均值
	float avg_iou = 0;
	float recall = 0;
	float recall75 = 0;
	float avg_cat = 0;
	//表示预测box包含对象与IOU好坏的评分
	float avg_obj = 0;
	//所有特征图上物体Po(预测机率与IOU正确率)的概率,有意义吗?
	float avg_anyobj = 0;
	//表示当前层与真实label正确配对的box数
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		//前面查找所有特征图(在这1个batch是n是三张）里的所有元素(19*19)里的所有confidence。
		//计算delta=期望输出-真实输出
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				//假设是主支，就是三个大框
				for (n = 0; n < l.n; ++n) {
					//box在特征图里的定位，每个特征图深度(4+1+l.classes)(4 box,1 Po,classes)概率
					int box_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 0);
					//对应预测的box
					box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
					float best_iou = 0;
					int best_t = 0;
					//每张图真实的标签，最多有max_boxes个box
					for (t = 0; t < l.max_boxes; ++t) {
						box truth = float_to_box(net.truth + t * (4 + 1) + b * l.truths, 1);
						if (!truth.x) break;
						float iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
							best_t = t;
						}
					}
					//拿到Po值索引
					int obj_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 4);
					avg_anyobj += l.output[obj_index];
					//先假定所有框都不正确，下面第二部分好的会重新设置
					l.delta[obj_index] = 0 - l.output[obj_index];
					//当前特征图与标签里的IOU满足ignore_thresh，l.delta置0
					if (best_iou > l.ignore_thresh) {
						l.delta[obj_index] = 0;
					}
					//truth_thresh几个配置都设的是1，best_iou能大于1?
					if (best_iou > l.truth_thresh) {
						l.delta[obj_index] = 1 - l.output[obj_index];
						int class = net.truth[best_t*(4 + 1) + b * l.truths + 4];
						if (l.map) class = l.map[class];
						int class_index = entry_index(l, b, n*l.w*l.h + j * l.w + i, 4 + 1);
						delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
						box truth = float_to_box(net.truth + best_t * (4 + 1) + b * l.truths, 1);
						delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);
					}
				}
			}
		}
		//每张图真实的标签，最多有max_boxes 个box
		for (t = 0; t < l.max_boxes; ++t) {
			box truth = float_to_box(net.truth + t * (4 + 1) + b * l.truths, 1);
			//一般只有1-4个
			if (!truth.x)
				break;
			float best_iou = 0;
			int best_n = 0;
			//找到格子索引(如在主支，就是(0-19]索引)
			i = (truth.x * l.w);
			j = (truth.y * l.h);
			box truth_shift = truth;
			truth_shift.x = truth_shift.y = 0;
			for (n = 0; n < l.total; ++n) {
				box pred = { 0 };
				pred.w = l.biases[2 * n] / net.w;
				pred.h = l.biases[2 * n + 1] / net.h;
				float iou = box_iou(pred, truth_shift);
				if (iou > best_iou) {
					best_iou = iou;
					best_n = n;
				}
			}
			//当前mask(3个)索引里包含best_n没(best_n一般在total[9]个索引中的一个)
			int mask_n = int_index(l.mask, best_n, l.n);
			//mask_n 在这表示是不是当前yolo层负责的n(3)个框之一
			if (mask_n >= 0) {
				//最准确的那个box索引
				int box_index = entry_index(l, b, mask_n*l.w*l.h + j * l.w + i, 0);
				//计算box的delta
				float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2 - truth.w*truth.h), l.w*l.h);
				//计算Po的delta
				int obj_index = entry_index(l, b, mask_n*l.w*l.h + j * l.w + i, 4);
				avg_obj += l.output[obj_index];
				l.delta[obj_index] = 1 - l.output[obj_index];
				//标签里的对应box的类别
				int class = net.truth[t*(4 + 1) + b * l.truths + 4];
				if (l.map) class = l.map[class];
				//预测对应的box类别索引
				int class_index = entry_index(l, b, mask_n*l.w*l.h + j * l.w + i, 4 + 1);
				//l.delta[class_index] 
				delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

				++count;
				++class_count;
				if (iou > .5) recall += 1;
				if (iou > .75) recall75 += 1;
				avg_iou += iou;
			}
		}
	}
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w*l.h*l.n*l.batch), recall / count, recall75 / count, count);
}

void backward_yolo_layer(const layer l, network net)
{
	//net.delta指向的是yolov3的上个卷积层l.delta
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((float)netw / w) < ((float)neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	}
	else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
		b.w *= (float)netw / new_w;
		b.h *= (float)neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

int yolo_num_detections(layer l, float thresh)
{
	int i, n;
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i) {
		for (n = 0; n < l.n; ++n) {
			//索引对应的confidence = P(object)* IOU
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			if (l.output[obj_index] > thresh) {
				++count;
			}
		}
	}
	return count;
}

void avg_flipped_yolo(layer l)
{
	int i, j, n, z;
	float *flip = l.output + l.outputs;
	for (j = 0; j < l.h; ++j) {
		for (i = 0; i < l.w / 2; ++i) {
			for (n = 0; n < l.n; ++n) {
				for (z = 0; z < l.classes + 4 + 1; ++z) {
					int i1 = z * l.w*l.h*l.n + n * l.w*l.h + j * l.w + i;
					int i2 = z * l.w*l.h*l.n + n * l.w*l.h + j * l.w + (l.w - i - 1);
					float swap = flip[i1];
					flip[i1] = flip[i2];
					flip[i2] = swap;
					if (z == 0) {
						flip[i1] = -flip[i1];
						flip[i2] = -flip[i2];
					}
				}
			}
		}
	}
	for (i = 0; i < l.outputs; ++i) {
		l.output[i] = (l.output[i] + flip[i]) / 2.;
	}
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
	int i, j, n;
	float *predictions = l.output;
	if (l.batch == 2) avg_flipped_yolo(l);
	int count = 0;
	for (i = 0; i < l.w*l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			//索引对应的confidence = P(object)* IOU
			int obj_index = entry_index(l, 0, n*l.w*l.h + i, 4);
			float objectness = predictions[obj_index];
			if (objectness <= thresh) continue;
			int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
			dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
			//索引对应的confidence = P(object)* IOU
			dets[count].objectness = objectness;
			dets[count].classes = l.classes;
			for (j = 0; j < l.classes; ++j) {
				int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
				float prob = objectness * predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
	return count;
}

#if GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
	copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	int b, n;
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n*l.w*l.h, 0);
			activate_array_gpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
			index = entry_index(l, b, n*l.w*l.h, 4);
			activate_array_gpu(l.output_gpu + index, (1 + l.classes)*l.w*l.h, LOGISTIC);
		}
	}
	if (!net.train || l.onlyforward) {
		cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
		return;
	}

	cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
	forward_yolo_layer(l, net);
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
	axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

