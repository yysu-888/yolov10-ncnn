#include "yolov10.h"

using namespace cv;
using namespace std;


Yolov10::Yolov10(std::string param_path,std::string model_path){

    yolo_net = make_unique<ncnn::Net>();

    yolo_net->opt.use_fp16_storage = true;
	yolo_net->opt.use_fp16_arithmetic = true;

    if (yolo_net->load_param(param_path.c_str()))
		exit(-1);
	if (yolo_net->load_model(model_path.c_str()))
		exit(-1);

}

int Yolov10::detect(const cv::Mat& bgr, std::vector<Object>& objects){
    
    int img_w = bgr.cols;
	int img_h = bgr.rows;

	int w = img_w;
	int h = img_h;
	float scale = 1.f;
	if (w > h)
	{
		scale = (float)target_size / w;
		w = target_size;
		h = h * scale;
	}
	else
	{
		scale = (float)target_size / h;
		h = target_size;
		w = w * scale;
	}

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);


    int wpad = target_size - w;
    int hpad = target_size - h;
    
	int top = hpad / 2;
	int bottom = hpad - hpad / 2;
	int left = wpad / 2;
	int right = wpad - wpad / 2;

	ncnn::Mat in_pad;
	ncnn::copy_make_border(in,
		in_pad,
		top,
		bottom,
		left,
		right,
		ncnn::BORDER_CONSTANT,
		114.f);

	const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	in_pad.substract_mean_normalize(0, norm_vals);

	ncnn::Extractor ex = yolo_net->create_extractor();

	ex.input("in0", in_pad);

	std::vector<Object> proposals;

	// stride 8 
	{
		ncnn::Mat out;
		ex.extract(stride[0].output_name.c_str(), out);

		std::vector<Object> objects8;
		generate_proposals(stride[0].stride, out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}

	// stride 16 
	{
		ncnn::Mat out;

		ex.extract(stride[1].output_name.c_str(), out);

		std::vector<Object> objects16;
		generate_proposals(stride[1].stride, out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}

	// stride 32 
	{
		ncnn::Mat out;

		ex.extract(stride[2].output_name.c_str(), out);
	

		std::vector<Object> objects32;
		generate_proposals(stride[2].stride, out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}
    // objects = proposals;
    for (auto& pro : proposals)
	{
        float x0 = pro.rect.x;
		float y0 = pro.rect.y;
		float x1 = pro.rect.x + pro.rect.width;
		float y1 = pro.rect.y + pro.rect.height;
		float& score = pro.prob;
		int& label = pro.label;

		x0 = (x0 - (wpad / 2)) / scale;
		y0 = (y0 - (hpad / 2)) / scale;
		x1 = (x1 - (wpad / 2)) / scale;
		y1 = (y1 - (hpad / 2)) / scale;

		x0 = clamp(x0, 0.f, img_w);
		y0 = clamp(y0, 0.f, img_h);
		x1 = clamp(x1, 0.f, img_w);
		y1 = clamp(y1, 0.f, img_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
		objects.push_back(obj);
	}

	return 0;
}


void Yolov10::non_max_suppression(std::vector<Object>& proposals,std::vector<Object>& results,
            int orin_h,int orin_w,float dh,
			float dw ,
			float ratio_h ,
			float ratio_w ,
			float conf_thres,
			float iou_thres){

	results.clear();

	for (auto& pro : proposals)
	{
        float x0 = pro.rect.x;
		float y0 = pro.rect.y;
		float x1 = pro.rect.x + pro.rect.width;
		float y1 = pro.rect.y + pro.rect.height;
		float& score = pro.prob;
		int& label = pro.label;

		x0 = (x0 - dw) / ratio_w;
		y0 = (y0 - dh) / ratio_h;
		x1 = (x1 - dw) / ratio_w;
		y1 = (y1 - dh) / ratio_h;

		x0 = clamp(x0, 0.f, orin_w);
		y0 = clamp(y0, 0.f, orin_h);
		x1 = clamp(x1, 0.f, orin_w);
		y1 = clamp(y1, 0.f, orin_h);

		Object obj;
		obj.rect.x = x0;
		obj.rect.y = y0;
		obj.rect.width = x1 - x0;
		obj.rect.height = y1 - y0;
		obj.prob = score;
		obj.label = label;
		results.push_back(obj);
	}
}


void Yolov10::generate_proposals(int stride,const ncnn::Mat& feat_blob,
	    const float prob_threshold,std::vector<Object>& objects){

	const int reg_max = 16;
	float dst[16];
	const int num_w = feat_blob.w;
	const int num_grid_y = feat_blob.c;
	const int num_grid_x = feat_blob.h;

	const int num_class = num_w - 4 * reg_max;

	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{

			const float* matat = feat_blob.channel(i).row(j);

			int class_index = 0;
			float class_score = -FLT_MAX;
			for (int c = 0; c < num_class; c++)
			{
				float score = matat[c];
				if (score > class_score)
				{
					class_index = c;
					class_score = score;
				}
			}
			if (class_score >= prob_threshold)
			{

				float x0 = j + 0.5f - softmax(matat + num_class, dst, 16);
				float y0 = i + 0.5f - softmax(matat + num_class + 16, dst, 16);
				float x1 = j + 0.5f + softmax(matat + num_class + 2 * 16, dst, 16);
				float y1 = i + 0.5f + softmax(matat + num_class + 3 * 16, dst, 16);

				x0 *= stride;
				y0 *= stride;
				x1 *= stride;
				y1 *= stride;

				Object obj;
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = x1 - x0;
				obj.rect.height = y1 - y0;
				obj.label = class_index;
				obj.prob = class_score;
				objects.push_back(obj);

			}
		}
	}
}


cv::Mat Yolov10::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects){
	static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		cv::rectangle(image, obj.rect, cv::Scalar(0, 0, 255),2);

		char text[256];
		sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}
	return image;
}


void Yolov10::detect_video(){
	cv::VideoCapture capture(0);

	while (true)
	{
		cv::Mat frame;
		capture >> frame;
		
		std::vector<Object> objects;

		TICK(yolov10);
		detect(frame, objects);
		TOCK(yolov10);

		cv::Mat image = draw_objects(frame, objects);

		cv::imshow("0", image);
		char c = cv::waitKey(1);
		if(c==27 || c==3) break;
	}
}

void Yolov10::detect_image(std::string img_path){
	
	cv::Mat m = cv::imread(img_path);
	std::vector<Object> objects;

	TICK(yolov10);
	detect(m, objects);
	TOCK(yolov10);
	
	cv::Mat image = draw_objects(m, objects);
	cv::imwrite("./detect.jpg",image);
	cv::imshow("0",image);
	cv::waitKey(0);
}