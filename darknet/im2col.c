#include "im2col.h"
#include <stdio.h>
float im2col_get_pixel(float *im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
     int channels,  int height,  int width,
     int ksize,  int stride, int pad, float* data_col) 
{
    int c,h,w;
	//当前层输出特征图的高度
    int height_col = (height + 2*pad - ksize) / stride + 1;
	//当前层输出特征图的宽度
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
	//c表示在当前层里卷积(一共channels个)里的索引
    for (c = 0; c < channels_col; ++c) {
		//对应总面积层的行方向索引
        int w_offset = c % ksize;
		//对应总面积层的列方向索引
        int h_offset = (c / ksize) % ksize;
		//卷积核的索引
        int c_im = c / ksize / ksize;
		//卷积核与输入特征得到中间层，方便后面做卷积计算
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                        im_row, im_col, c_im, pad);
            }
        }
    }
}

