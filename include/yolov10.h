#include "layer.h"
#include "net.h"
#include <float.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono>
#include "opencv2/opencv.hpp"

#define TICK(x) auto bench_##x=std::chrono::steady_clock::now();
#define TOCK(x) printf("%s:%lfs\n",#x,std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-bench_##x).count());

#define MAX_STRIDE 32

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};

typedef struct
{
    int stride;
    std::string output_name;
} stride_info;

static float softmax(const float* src,float* dst,int length)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}


static float clamp(float val,float min = 0.f,float max = 1280.f)
{
	return val > min ? (val < max ? val : max) : min;
}


class Yolov10{
public:

    Yolov10(std::string param_path,std::string model_path);
    void detect_video();
    void detect_image(std::string img_path);

private:
    void non_max_suppression(std::vector<Object>& proposals,std::vector<Object>& results,
            int orin_h,int orin_w,float dh = 0,float dw = 0,float ratio_h = 1.0f,float ratio_w = 1.0f,
            float conf_thres = 0.25f,float iou_thres = 0.65f);

    
    void generate_proposals(int stride,const ncnn::Mat& feat_blob,
	        const float prob_threshold,std::vector<Object>& objects);

    int detect(const cv::Mat& bgr, std::vector<Object>& objects);
    cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);


    std::unique_ptr<ncnn::Net> yolo_net;

    const int target_size = 640;
    const float prob_threshold = 0.25f;
	const float nms_threshold = 0.45f;

    std::vector<stride_info> stride=
    {
        {8,"out0"},
        {16,"out1"},
        {32,"out2"}
    };


}; // Yolov10