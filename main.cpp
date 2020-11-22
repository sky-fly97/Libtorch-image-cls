#include "Cls.h"

int main() {
	string model_path = "model_res18_new.pt";
	string class_filename = "classes.txt";
	Torch_Cls A(model_path, class_filename);

	while (true) {
		string img_path = "image_1";
		A.NetForward(img_path);
	}
	return 0;
}