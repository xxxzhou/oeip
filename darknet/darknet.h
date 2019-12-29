#pragma once

#ifndef DARKNET_API
#define DARKNET_API

#ifdef YOLO3_EXPORT
#define DARKNET_EXPORT __declspec(dllexport)
#else
#define DARKNET_EXPORT __declspec(dllimport)
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef PTHREAD
#include <pthread.h>
#endif 

#ifdef GPU
#define BLOCK 512

#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

#ifdef CUDNN
#include "cudnn.h"
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SECRET_NUM -1234
	extern int gpu_index;

	typedef struct {
		int classes;
		char **names;
	} metadata;

	metadata get_metadata(char *file);

	typedef struct {
		int *leaf;
		int n;
		int *parent;
		int *child;
		int *group;
		char **name;

		int groups;
		int *group_size;
		int *group_offset;
	} tree;
	tree *read_tree(char *filename);

	typedef enum {
		LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
	} ACTIVATION;

	typedef enum {
		PNG, BMP, TGA, JPG
	} IMTYPE;

	typedef enum {
		MULT, ADD, SUB, DIV
	} BINARY_ACTIVATION;

	typedef enum {
		CONVOLUTIONAL,
		DECONVOLUTIONAL,
		CONNECTED,
		MAXPOOL,
		SOFTMAX,
		DETECTION,
		DROPOUT,
		CROP,
		ROUTE,
		COST,
		NORMALIZATION,
		AVGPOOL,
		LOCAL,
		SHORTCUT,
		ACTIVE,
		RNN,
		GRU,
		LSTM,
		CRNN,
		BATCHNORM,
		NETWORK,
		XNOR,
		REGION,
		YOLO,
		ISEG,
		REORG,
		UPSAMPLE,
		LOGXENT,
		L2NORM,
		BLANK
	} LAYER_TYPE;

	typedef enum {
		SSE, MASKED, L1, SEG, SMOOTH, WGAN
	} COST_TYPE;

	typedef struct {
		int batch;
		float learning_rate;
		float momentum;
		float decay;
		int adam;
		float B1;
		float B2;
		float eps;
		int t;
	} update_args;

	struct network;
	typedef struct network network;

	struct layer;
	typedef struct layer layer;

	struct layer {
		LAYER_TYPE type;
		ACTIVATION activation;
		COST_TYPE cost_type;
		//预测方法
		void(*forward)   (struct layer, struct network);
		//训练，损失函数确定方向
		void(*backward)  (struct layer, struct network);
		//训练，根据方向更新参数
		void(*update)    (struct layer, update_args);
		void(*forward_gpu)   (struct layer, struct network);
		void(*backward_gpu)  (struct layer, struct network);
		void(*update_gpu)    (struct layer, update_args);
		//是否正则化
		int batch_normalize;
		int shortcut;
		//每次训练多少图片
		int batch;
		int forced;
		//当前层权重参数是否需要翻转,如要求的权重格式是 输出特征数(列)*(输入特征*SIZE*SIZE)(行)
		//如果权重格式变成 (输入特征*SIZE*SIZE)(列)*输出特征数(行),就需要翻转
		int flipped;
		//当前层batch为1时所有输入特征图包含的元素个数，每个元素为一个float
		int inputs;
		//当前层batch为1的所有输出特征图包含的元素个数，每个元素为一个float
		int outputs;
		//当前层卷积核的参数量(c*n*size*size)，每个元素为一个float
		int nweights;
		//当前层偏置的参数量(n)，每个元素为一个float
		int nbiases;
		int extra;
		//对应训练的标签数值
		int truths;
		//每层输入特征图要求和长/宽/个数
		int h, w, c;
		//每层输出特征图要求和长/宽/个数，可以根据卷积核参数与输入求得对应数据
		int out_h, out_w, out_c;
		//cfg文件卷积层中的filters，对应经过当前卷积层后输出的特征图个数
		int n;
		int max_boxes;
		int groups;
		//卷积核的大小
		int size;
		//DETECTION层所用，代表输出特征图上的格子数
		int side;
		//卷积核的步长
		int stride;
		int reverse;
		int flatten;
		int spatial;
		//卷积核是否自动设置padding
		int pad;
		int sqrt;
		//在裁剪层确定是否左右是否翻转
		int flip;
		int index;
		int binary;
		int xnor;
		int steps;
		int hidden;
		int truth;
		float smooth;
		float dot;
		float angle;
		float jitter;
		float saturation;
		float exposure;
		float shift;
		float ratio;
		float learning_rate_scale;
		float clip;
		int noloss;
		int softmax;
		int classes;
		//坐标参数 四个
		int coords;
		//暂时来看,ylolv2检测层专用
		int background;
		int rescore;
		int objectness;
		int joint;
		//裁剪层是否调整,如果不调整，裁剪输出层对应输入层原值，否则 x*2-1
		int noadjust;
		int reorg;
		int log;
		int tanh;
		//yolov3里分组检测索引，一般分成[0,1,2/3,4,5/6,7,8]
		int *mask;
		//yolov3里3组，每组3个anchor,一共9个
		int total;

		float alpha;
		float beta;
		float kappa;

		float coord_scale;
		float object_scale;
		float noobject_scale;
		float mask_scale;
		float class_scale;
		int bias_match;
		int random;
		float ignore_thresh;
		float truth_thresh;
		float thresh;
		float focus;
		int classfix;
		int absolute;

		int onlyforward;
		int stopbackward;
		int dontload;
		int dontsave;
		int dontloadscales;
		int numload;

		float temperature;
		float probability;
		float scale;

		char  * cweights;
		int   * indexes;
		int   * input_layers;
		int   * input_sizes;
		int   * map;
		int   * counts;
		float ** sums;
		float * rand;
		//一般检测层使用,比较真实与预测的差距,cost层可以预测时会统计所有误差
		float * cost;
		float * state;
		float * prev_state;
		float * forgot_state;
		float * forgot_delta;
		float * state_delta;
		float * combine_cpu;
		float * combine_delta_cpu;

		float * concat;
		float * concat_delta;
		//二值化的weight
		float * binary_weights;
		//当前层偏置参数，同当前层输出特征图通道数n,当前层正规化需要的参数,保存在参数集中
		//yolo2中表明的是对应的层anchors参数,这个个数对应yolo2层n*2,表明每个box用二个值
		float * biases;
		//偏置的梯度=-2*delta(期望输出-真实输出)
		float * bias_updates;

		//缩放,同当前层输出特征图通道数n,当前层正规化需要的参数,保存在参数集中
		float * scales;
		float * scale_updates;
		//当前层权重参数，大小等同当前n*c*size*size
		float * weights;
		//训练参数的梯度=delta(期望输出-预测输出)*输入
		float * weight_updates;
		//训练,保存 期望输出-预测输出 的值 W(new)=W(old)+2*学习率*输入*(期望输出-预测输出)
		float * delta;
		//每层输出的特征图数据
		float * output;
		//损失,一般是正值
		float * loss;
		float * squared;
		float * norms;
		//训练用的
		float * spatial_mean;
		//训练,求得平均值
		float * mean;
		float * variance;

		float * mean_delta;
		float * variance_delta;
		//均值,同当前层输出特征图通道数n,当前层正规化需要的参数,保存在参数集中
		float * rolling_mean;
		//方差,同当前层输出特征图通道数n,当前层正规化需要的参数,保存在参数集中
		float * rolling_variance;
		//正规化时，把经过卷积的输出结果保存下来用于训练
		float * x;
		float * x_norm;

		//在学习率为adam模式下，需要的记录的变量
		float * m;
		float * v;
		float * bias_m;
		float * bias_v;
		float * scale_m;
		float * scale_v;


		float *z_cpu;
		float *r_cpu;
		float *h_cpu;
		float * prev_state_cpu;

		float *temp_cpu;
		float *temp2_cpu;
		float *temp3_cpu;

		float *dh_cpu;
		float *hh_cpu;
		float *prev_cell_cpu;
		float *cell_cpu;
		float *f_cpu;
		float *i_cpu;
		float *g_cpu;
		float *o_cpu;
		float *c_cpu;
		float *dc_cpu;

		float * binary_input;

		struct layer *input_layer;
		struct layer *self_layer;
		struct layer *output_layer;

		struct layer *reset_layer;
		struct layer *update_layer;
		struct layer *state_layer;

		struct layer *input_gate_layer;
		struct layer *state_gate_layer;
		struct layer *input_save_layer;
		struct layer *state_save_layer;
		struct layer *input_state_layer;
		struct layer *state_state_layer;

		struct layer *input_z_layer;
		struct layer *state_z_layer;

		struct layer *input_r_layer;
		struct layer *state_r_layer;

		struct layer *input_h_layer;
		struct layer *state_h_layer;

		struct layer *wz;
		struct layer *uz;
		struct layer *wr;
		struct layer *ur;
		struct layer *wh;
		struct layer *uh;
		struct layer *uo;
		struct layer *wo;
		struct layer *uf;
		struct layer *wf;
		struct layer *ui;
		struct layer *wi;
		struct layer *ug;
		struct layer *wg;

		tree *softmax_tree;

		size_t workspace_size;

#ifdef GPU
		int *indexes_gpu;

		float *z_gpu;
		float *r_gpu;
		float *h_gpu;

		float *temp_gpu;
		float *temp2_gpu;
		float *temp3_gpu;

		float *dh_gpu;
		float *hh_gpu;
		float *prev_cell_gpu;
		float *cell_gpu;
		float *f_gpu;
		float *i_gpu;
		float *g_gpu;
		float *o_gpu;
		float *c_gpu;
		float *dc_gpu;

		float *m_gpu;
		float *v_gpu;
		float *bias_m_gpu;
		float *scale_m_gpu;
		float *bias_v_gpu;
		float *scale_v_gpu;

		float * combine_gpu;
		float * combine_delta_gpu;

		float * prev_state_gpu;
		float * forgot_state_gpu;
		float * forgot_delta_gpu;
		float * state_gpu;
		float * state_delta_gpu;
		float * gate_gpu;
		float * gate_delta_gpu;
		float * save_gpu;
		float * save_delta_gpu;
		float * concat_gpu;
		float * concat_delta_gpu;

		float * binary_input_gpu;
		float * binary_weights_gpu;

		float * mean_gpu;
		float * variance_gpu;

		float * rolling_mean_gpu;
		float * rolling_variance_gpu;

		float * variance_delta_gpu;
		float * mean_delta_gpu;

		float * x_gpu;
		float * x_norm_gpu;
		float * weights_gpu;
		float * weight_updates_gpu;
		float * weight_change_gpu;

		float * biases_gpu;
		float * bias_updates_gpu;
		float * bias_change_gpu;

		float * scales_gpu;
		float * scale_updates_gpu;
		float * scale_change_gpu;

		float * output_gpu;
		float * loss_gpu;
		float * delta_gpu;
		float * rand_gpu;
		float * squared_gpu;
		float * norms_gpu;
#ifdef CUDNN
		cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
		cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
		cudnnTensorDescriptor_t normTensorDesc;
		cudnnFilterDescriptor_t weightDesc;
		cudnnFilterDescriptor_t dweightDesc;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnConvolutionFwdAlgo_t fw_algo;
		cudnnConvolutionBwdDataAlgo_t bd_algo;
		cudnnConvolutionBwdFilterAlgo_t bf_algo;
#endif
#endif
	};

	void free_layer(layer);

	typedef enum {
		CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
	} learning_rate_policy;

	//默认值请看parse_net_options 实现
	typedef struct network {
		int n;
		int batch;
		//已经训练完成数
		size_t *seen;
		int *t;
		float epoch;
		//用于一次加载subdivisions*batch个数据
		int subdivisions;
		layer *layers;
		float *output;
		learning_rate_policy policy;

		float learning_rate;
		float momentum;
		float decay;
		float gamma;
		float scale;
		float power;
		int time_steps;
		int step;
		int max_batches;
		float *scales;
		int   *steps;
		int num_steps;
		int burn_in;

		int adam;
		float B1;
		float B2;
		float eps;
		//一个batch中,网络最开始要输入的给第一层特征图的所有元素(n*h*w)
		int inputs;
		//记录网络最后输出元素数
		int outputs;
		//标签数据长度，一般等于网络最后层输出长度
		int truths;
		int notruth;
		int h, w, c;
		int max_crop;
		int min_crop;
		float max_ratio;
		float min_ratio;
		int center;
		float angle;
		float aspect;
		float exposure;
		float saturation;
		float hue;
		int random;

		int gpu_index;
		tree *hierarchy;
		//在预测时,保存上层的输出结果
		float *input;
		//标签数据
		float *truth;
		//反向传播梯度，从尾到头梯度乘积，当前层与
		float *delta;
		//中间层,一般用来在卷积层方便计算保存输入层转化的中间层。
		//特点是当前层输入特征数*输出特征图长*输出特征图宽
		float *workspace;
		//是否在训练
		int train;
		int index;
		float *cost;
		float clip;

#ifdef GPU
		float *input_gpu;
		float *truth_gpu;
		float *delta_gpu;
		float *output_gpu;
#endif

	} network;

	typedef struct {
		int w;
		int h;
		float scale;
		float rad;
		float dx;
		float dy;
		float aspect;
	} augment_args;

	typedef struct {
		int w;
		int h;
		int c;
		float *data;
	} image;

	typedef struct {
		float x, y, w, h;
	} box;

	typedef struct detection {
		box bbox;
		int classes;
		//对应所有classes的概率
		float *prob;
		float *mask;
		//模型对框中包含对象的信心与对应box正确率,Po*IOU
		float objectness;
		int sort_class;
	} detection;
	//matrix用来表示特征图的集合
	typedef struct matrix {
		//rows表示特征图的个数，cols表示特征图的大小
		int rows, cols;
		float **vals;
	} matrix;


	typedef struct data {
		//特征图的长宽
		int w, h;
		//训练集
		matrix X;
		//标签集
		matrix y;
		int shallow;
		int *num_boxes;
		box **boxes;
	} data;

	typedef enum {
		CLASSIFICATION_DATA, DETECTION_DATA, CAPTCHA_DATA, REGION_DATA, IMAGE_DATA, COMPARE_DATA, WRITING_DATA, SWAG_DATA, TAG_DATA, OLD_CLASSIFICATION_DATA, STUDY_DATA, DET_DATA, SUPER_DATA, LETTERBOX_DATA, REGRESSION_DATA, SEGMENTATION_DATA, INSTANCE_DATA, ISEG_DATA
	} data_type;

	typedef struct load_args {
		int threads;
		char **paths;
		char *path;
		int n;
		int m;
		char **labels;
		int h;
		int w;
		int out_w;
		int out_h;
		int nh;
		int nw;
		int num_boxes;
		int min, max, size;
		int classes;
		int background;
		int scale;
		int center;
		int coords;
		float jitter;
		float angle;
		float aspect;
		float saturation;
		float exposure;
		float hue;
		data *d;
		image *im;
		image *resized;
		data_type type;
		tree *hierarchy;
	} load_args;

	typedef struct {
		int id;
		float x, y, w, h;
		float left, right, top, bottom;
	} box_label;

	typedef struct node {
		void *val;
		struct node *next;
		struct node *prev;
	} node;

	typedef struct list {
		int size;
		node *front;
		node *back;
	} list;

	DARKNET_EXPORT network *load_network(char *cfg, char *weights, int clear);
	DARKNET_EXPORT load_args get_base_args(network *net);

	DARKNET_EXPORT void free_data(data d);
#ifdef PTHREAD
	DARKNET_EXPORT pthread_t load_data(load_args args);
#endif
	DARKNET_EXPORT list *read_data_cfg(char *filename);
	DARKNET_EXPORT list *read_cfg(char *filename);
	DARKNET_EXPORT unsigned char *read_file(char *filename);
	DARKNET_EXPORT data resize_data(data orig, int w, int h);
	DARKNET_EXPORT data *tile_data(data orig, int divs, int size);
	DARKNET_EXPORT data select_data(data *orig, int *inds);

	DARKNET_EXPORT void forward_network(network *net);
	DARKNET_EXPORT void backward_network(network *net);
	DARKNET_EXPORT void update_network(network *net);


	DARKNET_EXPORT float dot_cpu(int N, float *X, int INCX, float *Y, int INCY);
	DARKNET_EXPORT void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
	DARKNET_EXPORT void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
	DARKNET_EXPORT void scal_cpu(int N, float ALPHA, float *X, int INCX);
	DARKNET_EXPORT void fill_cpu(int N, float ALPHA, float * X, int INCX);
	DARKNET_EXPORT void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
	DARKNET_EXPORT void softmax(float *input, int n, float temp, int stride, float *output);

	DARKNET_EXPORT int best_3d_shift_r(image a, image b, int min, int max);
#ifdef GPU
	DARKNET_EXPORT void axpy_gpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY);
	DARKNET_EXPORT void fill_gpu(int N, float ALPHA, float * X, int INCX);
	DARKNET_EXPORT void scal_gpu(int N, float ALPHA, float * X, int INCX);
	DARKNET_EXPORT void copy_gpu(int N, float * X, int INCX, float * Y, int INCY);

	DARKNET_EXPORT void cuda_set_device(int n);
	DARKNET_EXPORT void cuda_free(float *x_gpu);
	DARKNET_EXPORT float *cuda_make_array(float *x, size_t n);
	DARKNET_EXPORT void cuda_pull_array(float *x_gpu, float *x, size_t n);
	DARKNET_EXPORT float cuda_mag_array(float *x_gpu, size_t n);
	DARKNET_EXPORT void cuda_push_array(float *x_gpu, float *x, size_t n);

	DARKNET_EXPORT void forward_network_gpu(network *net);
	DARKNET_EXPORT void backward_network_gpu(network *net);
	DARKNET_EXPORT void update_network_gpu(network *net);

	DARKNET_EXPORT float train_networks(network **nets, int n, data d, int interval);
	DARKNET_EXPORT void sync_nets(network **nets, int n, int interval);
	DARKNET_EXPORT void harmless_update_network_gpu(network *net);

	DARKNET_EXPORT void forward_network_gpudata(network *netp);	
#endif
	DARKNET_EXPORT void network_predict_gpudata(network *net, float *input);
	DARKNET_EXPORT image get_label(image **characters, char *string, int size);
	DARKNET_EXPORT void draw_label(image a, int r, int c, image label, const float *rgb);
	DARKNET_EXPORT void save_image(image im, const char *name);
	DARKNET_EXPORT void save_image_options(image im, const char *name, IMTYPE f, int quality);
	DARKNET_EXPORT void get_next_batch(data d, int n, int offset, float *X, float *y);
	DARKNET_EXPORT void grayscale_image_3c(image im);
	DARKNET_EXPORT void normalize_image(image p);
	DARKNET_EXPORT void matrix_to_csv(matrix m);
	DARKNET_EXPORT float train_network_sgd(network *net, data d, int n);
	DARKNET_EXPORT void rgbgr_image(image im);
	DARKNET_EXPORT data copy_data(data d);
	DARKNET_EXPORT data concat_data(data d1, data d2);
	DARKNET_EXPORT data load_cifar10_data(char *filename);
	DARKNET_EXPORT float matrix_topk_accuracy(matrix truth, matrix guess, int k);
	DARKNET_EXPORT void matrix_add_matrix(matrix from, matrix to);
	DARKNET_EXPORT void scale_matrix(matrix m, float scale);
	DARKNET_EXPORT matrix csv_to_matrix(char *filename);
	DARKNET_EXPORT float *network_accuracies(network *net, data d, int n);
	DARKNET_EXPORT float train_network_datum(network *net);
	DARKNET_EXPORT image make_random_image(int w, int h, int c);

	DARKNET_EXPORT void denormalize_connected_layer(layer l);
	DARKNET_EXPORT void denormalize_convolutional_layer(layer l);
	DARKNET_EXPORT void statistics_connected_layer(layer l);
	DARKNET_EXPORT void rescale_weights(layer l, float scale, float trans);
	DARKNET_EXPORT void rgbgr_weights(layer l);
	DARKNET_EXPORT image *get_weights(layer l);

	DARKNET_EXPORT void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix, int avg, float hier_thresh, int w, int h, int fps, int fullscreen);
	DARKNET_EXPORT void get_detection_detections(layer l, int w, int h, float thresh, detection *dets);

	DARKNET_EXPORT char *option_find_str(list *l, char *key, char *def);
	DARKNET_EXPORT int option_find_int(list *l, char *key, int def);
	DARKNET_EXPORT int option_find_int_quiet(list *l, char *key, int def);

	DARKNET_EXPORT network *parse_network_cfg(char *filename);
	DARKNET_EXPORT void save_weights(network *net, char *filename);
	DARKNET_EXPORT void load_weights(network *net, char *filename);
	DARKNET_EXPORT void save_weights_upto(network *net, char *filename, int cutoff);
	DARKNET_EXPORT void load_weights_upto(network *net, char *filename, int start, int cutoff);

	DARKNET_EXPORT void zero_objectness(layer l);
	DARKNET_EXPORT void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets);
	DARKNET_EXPORT int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets);
	DARKNET_EXPORT void free_network(network *net);
	DARKNET_EXPORT void set_batch_network(network *net, int b);
	DARKNET_EXPORT void set_temp_network(network *net, float t);
	DARKNET_EXPORT image load_image(char *filename, int w, int h, int c);
	DARKNET_EXPORT image load_image_color(char *filename, int w, int h);
	DARKNET_EXPORT image make_image(int w, int h, int c);
	DARKNET_EXPORT image resize_image(image im, int w, int h);
	DARKNET_EXPORT void censor_image(image im, int dx, int dy, int w, int h);
	DARKNET_EXPORT image letterbox_image(image im, int w, int h);
	DARKNET_EXPORT image crop_image(image im, int dx, int dy, int w, int h);
	DARKNET_EXPORT image center_crop_image(image im, int w, int h);
	DARKNET_EXPORT image resize_min(image im, int min);
	DARKNET_EXPORT image resize_max(image im, int max);
	DARKNET_EXPORT image threshold_image(image im, float thresh);
	DARKNET_EXPORT image mask_to_rgb(image mask);
	DARKNET_EXPORT int resize_network(network *net, int w, int h);
	DARKNET_EXPORT void free_matrix(matrix m);
	DARKNET_EXPORT void test_resize(char *filename);
	DARKNET_EXPORT int show_image(image p, const char *name, int ms);
	DARKNET_EXPORT image copy_image(image p);
	DARKNET_EXPORT void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
	DARKNET_EXPORT float get_current_rate(network *net);
	DARKNET_EXPORT void composite_3d(char *f1, char *f2, char *out, int delta);
	DARKNET_EXPORT data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h);
	DARKNET_EXPORT size_t get_current_batch(network *net);
	DARKNET_EXPORT void constrain_image(image im);
	DARKNET_EXPORT image get_network_image_layer(network *net, int i);
	DARKNET_EXPORT layer get_network_output_layer(network *net);
	DARKNET_EXPORT void top_predictions(network *net, int n, int *index);
	DARKNET_EXPORT void flip_image(image a);
	DARKNET_EXPORT void vflip_image(image a);
	DARKNET_EXPORT image float_to_image(int w, int h, int c, float *data);
	DARKNET_EXPORT void ghost_image(image source, image dest, int dx, int dy);
	DARKNET_EXPORT float network_accuracy(network *net, data d);
	DARKNET_EXPORT void random_distort_image(image im, float hue, float saturation, float exposure);
	DARKNET_EXPORT void fill_image(image m, float s);
	DARKNET_EXPORT image grayscale_image(image im);
	DARKNET_EXPORT void rotate_image_cw(image im, int times);
	DARKNET_EXPORT double what_time_is_it_now();
	DARKNET_EXPORT image rotate_image(image m, float rad);
	DARKNET_EXPORT void visualize_network(network *net);
	DARKNET_EXPORT float box_iou(box a, box b);
	DARKNET_EXPORT data load_all_cifar10();
	DARKNET_EXPORT box_label *read_boxes(char *filename, int *n);
	DARKNET_EXPORT box float_to_box(float *f, int stride);
	DARKNET_EXPORT void draw_detections(image im, detection *dets, int num, float thresh, char **names, image **alphabet, int classes);

	DARKNET_EXPORT matrix network_predict_data(network *net, data test);
	DARKNET_EXPORT image **load_alphabet();
	DARKNET_EXPORT image get_network_image(network *net);
	DARKNET_EXPORT float *network_predict(network *net, float *input);

	DARKNET_EXPORT int network_width(network *net);
	DARKNET_EXPORT int network_height(network *net);
	DARKNET_EXPORT float *network_predict_image(network *net, image im);
	DARKNET_EXPORT void network_detect(network *net, image im, float thresh, float hier_thresh, float nms, detection *dets);
	DARKNET_EXPORT detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num);
	DARKNET_EXPORT void free_detections(detection *dets, int n);

	DARKNET_EXPORT void reset_network_state(network *net, int b);

	DARKNET_EXPORT char **get_labels(char *filename);
	DARKNET_EXPORT void do_nms_obj(detection *dets, int total, int classes, float thresh);
	DARKNET_EXPORT void do_nms_sort(detection *dets, int total, int classes, float thresh);

	DARKNET_EXPORT matrix make_matrix(int rows, int cols);

#ifdef OPENCV
	DARKNET_EXPORT void *open_video_stream(const char *f, int c, int w, int h, int fps);
	DARKNET_EXPORT image get_image_from_stream(void *p);
	DARKNET_EXPORT void make_window(char *name, int w, int h, int fullscreen);
#endif

	DARKNET_EXPORT void free_image(image m);
	DARKNET_EXPORT float train_network(network *net, data d);
#ifdef PTHREAD
	DARKNET_EXPORT pthread_t load_data_in_thread(load_args args);
	DARKNET_EXPORT void load_data_blocking(load_args args);
#endif
	DARKNET_EXPORT list *get_paths(char *filename);
	DARKNET_EXPORT void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
	DARKNET_EXPORT void change_leaves(tree *t, char *leaf_list);

	DARKNET_EXPORT int find_int_arg(int argc, char **argv, char *arg, int def);
	DARKNET_EXPORT float find_float_arg(int argc, char **argv, char *arg, float def);
	DARKNET_EXPORT int find_arg(int argc, char* argv[], char *arg);
	DARKNET_EXPORT char *find_char_arg(int argc, char **argv, char *arg, char *def);
	DARKNET_EXPORT char *basecfg(char *cfgfile);
	DARKNET_EXPORT void find_replace(char *str, char *orig, char *rep, char *output);
	DARKNET_EXPORT void free_ptrs(void **ptrs, int n);
	DARKNET_EXPORT char *fgetl(FILE *fp);
	DARKNET_EXPORT void strip(char *s);
	DARKNET_EXPORT float sec(clock_t clocks);
	DARKNET_EXPORT void **list_to_array(list *l);
	DARKNET_EXPORT void top_k(float *a, int n, int k, int *index);
	DARKNET_EXPORT int *read_map(char *filename);
	DARKNET_EXPORT void error(const char *s);
	DARKNET_EXPORT int max_index(float *a, int n);
	DARKNET_EXPORT int max_int_index(int *a, int n);
	DARKNET_EXPORT int sample_array(float *a, int n);
	DARKNET_EXPORT int *random_index_order(int min, int max);
	DARKNET_EXPORT void free_list(list *l);
	DARKNET_EXPORT float mse_array(float *a, int n);
	DARKNET_EXPORT float variance_array(float *a, int n);
	DARKNET_EXPORT float mag_array(float *a, int n);
	DARKNET_EXPORT void scale_array(float *a, int n, float s);
	DARKNET_EXPORT float mean_array(float *a, int n);
	DARKNET_EXPORT float sum_array(float *a, int n);
	DARKNET_EXPORT void normalize_array(float *a, int n);
	DARKNET_EXPORT int *read_intlist(char *s, int *n, int d);
	DARKNET_EXPORT size_t rand_size_t();
	DARKNET_EXPORT float rand_normal();
	DARKNET_EXPORT float rand_uniform(float min, float max);

	DARKNET_EXPORT data load_data_region(int n, char **paths, int m, int w, int h, int size, int classes, float jitter, float hue, float saturation, float exposure);
	DARKNET_EXPORT data load_data_detection(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);
	DARKNET_EXPORT data load_data_autoparamet(int n, char **paths, int m, int w, int h, int boxes, int classes, float jitter, float hue, float saturation, float exposure);

#ifdef __cplusplus
}
#endif
#endif
