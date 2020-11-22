#include <iostream>
#include <io.h>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr int NUM_CLASSES = 2;

const cv::Scalar colors[] = {
	{0, 255, 255},
	{255, 255, 0},
	{0, 255, 0},
	{255, 0, 0}
};

const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

const float mean_[3] = { 0.485, 0.456, 0.406 };
const float std_[3] = { 0.229, 0.224, 0.225 };

class Torch_Cls {
private:
	int image_nums = 11;
	int image_h = 224;
	int image_w = 224;
	vector<string> image_files;
	vector<cv::Mat> image_group;
	vector<std::string> class_names;
	torch::jit::script::Module module;
	cv::Mat image, image_out;
	torch::Tensor img_tensor;
	//torch::Tensor data_output;
	torch::Tensor output;
public:
	Torch_Cls(string&, string&);
	~Torch_Cls() {};
	void loadModel(string& model_path);
	void imageProcess(string& img_path);
	void NetForward(string& img_path);
	void readClassNames(string& class_filename);
	void getFiles(string& path);
};
