#include "yolov10.h"

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s [detect mode 0-image 1-video]\n", argv[0]);
		return -1;
	}

    std::string param_path="./models/yolov10n.param";
    std::string model_path="./models/yolov10n.bin";

    Yolov10 yolov10(param_path,model_path);

	if(atoi(argv[1]) == 0){
		if(argc !=3){
			fprintf(stderr, "Usage: %s [detect_mode image_path]\n", argv[0]);
			return -1;
		}
		std::string img_path = argv[2];
       
		yolov10.detect_image(img_path);
	}
	else if (atoi(argv[1]) == 1){
		yolov10.detect_video();
	}
    
	return 0;
}