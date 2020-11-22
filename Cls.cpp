#include "Cls.h"

Torch_Cls::Torch_Cls(string& model_path, string& class_filename)
{
	image_files.clear();
	image_group.clear();
	loadModel(model_path);
	readClassNames(class_filename);
}

//Find all images
void Torch_Cls::getFiles(string& path)
{
	intptr_t hFile = 0;//文件句柄，过会儿用来查找
	struct _finddata_t fileinfo;//文件信息
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		//如果查找到第一个文件
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))//如果是文件夹
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name));
			}
			else//如果是文件
			{
				image_files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);	//能寻找到其他文件

		_findclose(hFile);	//结束查找，关闭句柄
	}
}

//从标签文件读取类别
void Torch_Cls::readClassNames(string& filename)
{
	std::ifstream class_file(filename);
	if (!class_file)
	{
		std::cerr << "failed to open classes.txt\n";
		exit(-1);
	}
	std::string line;
	while (std::getline(class_file, line))
		class_names.push_back(line);
}

//load model
void Torch_Cls::loadModel(string& model_path) {
	std::cout << "----------------Model Loading----------------" << std::endl;
	auto load_start = std::chrono::steady_clock::now();
	this->module = torch::jit::load(model_path);
	this->module.to(torch::kCUDA);
	this->module.eval();
	auto load_end = std::chrono::steady_clock::now();
	float load_time = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
	printf("\n loading model cost : %.2f (ms)", load_time);
}

// 预处理
void Torch_Cls::imageProcess(string& img_path) {
	this->image = cv::imread(img_path);
	this->image_out = this->image.clone();
	cv::resize(this->image_out, this->image_out, cv::Size(224, 224), cv::INTER_LINEAR);
	cv::cvtColor(this->image_out, this->image_out, cv::COLOR_BGR2RGB);
	this->image_out.convertTo(this->image_out, CV_32F, 1.0 / 255);
	this->img_tensor = torch::from_blob(this->image_out.data, { 1, this->image_out.rows, this->image_out.cols, 3 }, torch::kFloat32); //(1,h,w,3)
	this->img_tensor = this->img_tensor.permute({ 0, 3, 1, 2 }); //(1, 3, h, w)
	//img_tensor = img_tensor.div(255);
	this->img_tensor[0][0] = this->img_tensor[0][0].sub_(0.485).div_(0.229);
	this->img_tensor[0][1] = this->img_tensor[0][1].sub_(0.456).div_(0.224);
	this->img_tensor[0][2] = this->img_tensor[0][2].sub_(0.406).div_(0.225);
}

// 前向推理
void Torch_Cls::NetForward(string& img_path) {
	auto process_start = std::chrono::steady_clock::now();
	getFiles(img_path);
	float max_index, max_score;
	for (int i = 0; i < image_nums; i++)
	{
		imageProcess(image_files[i]);
		torch::NoGradGuard no_grad;
		std::vector<torch::jit::IValue> inputs;
		inputs.emplace_back(this->img_tensor.to(torch::kCUDA));
		this->output = this->module.forward(inputs).toTensor();
		//std::cout << this->output << std::endl;
		auto max_result = this->output.max(1, true);
	    max_score = std::get<0>(max_result).item<float>();
	    max_index = std::get<1>(max_result).item<float>();
		if (max_index == 0) {
			std::cout << image_files[i] << std::endl;
			printf("\n current image classification : %s, possible : %.2f\n", this->class_names[max_index], max_score);
			break;
		}
	}
	auto process_end = std::chrono::steady_clock::now();
	float total_time = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start).count();
	//Result show
	float inference_fps = 1000.0 / total_time;
	std::ostringstream stats_ss;
	stats_ss << std::fixed << std::setprecision(2);
	stats_ss << "Inference FPS: " << inference_fps << ",  result : " << this->class_names[max_index] << ", possible: " << max_score;
	std::cout << "Inference FPS: " << inference_fps << std::endl;
	
	//auto stats = stats_ss.str();
	//int baseline;
	//auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.4, 0.4, &baseline);
	//cv::rectangle(this->image, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
	//cv::putText(this->image, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.4, cv::Scalar(255, 255, 255));
	//cv::imwrite("infer.jpg", this->image);
	//cv::namedWindow("output");
	//cv::imshow("output", this->image);
	//cv::waitKey();
}