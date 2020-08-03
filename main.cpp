#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include "utils.cpp"
#include <time.h>

#define IOU_THRES 0.3
#define CONF_THRES 0.4


std::vector<std::string> LoadClassNames(const std::string& path) {
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline(infile, line)) {
            class_names.emplace_back(line);
        }

        infile.close();
    }
    else {
        std::cerr << "Errir loading the class names!\n";
    }
    return class_names;
}

int main() {

    torch::jit::script::Module module = torch::jit::load("yolov5s.torchscript.pt");
    module.eval();

//    clock_t  start = clock();

//    timespec t1, t2;
//    clock_gettime(CLOCK_MONOTONIC, &t1);
    const auto t1 = std::chrono::system_clock::now();
    cv::Mat img = cv::imread("zidane.jpg");
//    cv::imshow("input", img);
//    cv::waitKey(0);

    cv::Mat img_input = img.clone();
    std::vector<float> padding_info = imagePadding(img_input, img_input, cv::Size(640, 640));
    const float pad_w = padding_info[0];
    const float pad_h = padding_info[1];
    const float scale = padding_info[2];

    cv::cvtColor(img_input, img_input, cv::COLOR_BGR2RGB);
    img_input.convertTo(img_input, CV_32FC3, 1.0f / 255.0f);

    torch::Tensor input_tensor = torch::from_blob(img_input.data, {1, img_input.rows, img_input.cols, img_input.channels()});
    input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(input_tensor);
//    std::cout << input_tensor.sizes() << std::endl;

    torch::jit::IValue pred = module.forward(inputs);
    auto detections = pred.toTuple()->elements()[0].toTensor();

    auto result = PostProcessing(detections, CONF_THRES, IOU_THRES);

//    std::cout << result.sizes() << std::endl;

    auto idx_mask = result * (result.select(1, 0) == 0).to(torch::kFloat32).unsqueeze(1);
    auto idx_mask_index = torch::nonzero(idx_mask.select(1, 1)).squeeze();
    const auto& result_data_demo = result.index_select(0, idx_mask_index).slice(1, 1, 7);

    const auto& demo_data = result_data_demo.accessor<float, 2>();
//    std::cout << result << std::endl;

//    clock_t end = clock();
//    clock_gettime(CLOCK_MONOTONIC, &t2);

//    std::vector<std::tuple<cv::Rect, float, int>> demo_data_vec;
    auto class_names = LoadClassNames("coco.names");
    for (int i = 0; i < result.size(0); ++i) {
        auto x1 = static_cast<int>((demo_data[i][0] - pad_w) / scale);
        auto y1 = static_cast<int>((demo_data[i][1] - pad_h) / scale);
        auto x2 = static_cast<int>((demo_data[i][2] - pad_w) / scale);
        auto y2 = static_cast<int>((demo_data[i][3] - pad_h) / scale);
        auto class_index = static_cast<int>(demo_data[i][5]);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << demo_data[i][4];
        std::string str = class_names[class_index] + " " + ss.str();

        cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
        cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
        cv::putText(img, str, cv::Point(rect.tl().x, rect.tl().y - 6),
                    cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 0, 255), 2);
        //        std::tuple<cv::Rect, float, int> t = std::make_tuple(rect, demo_data[i][4], demo_data[i][5]);
//        demo_data_vec.emplace_back(t);



    }

    const auto t2 = std::chrono::system_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Time: " << duration * 1e-3 << std::endl;
    cv::imshow("img", img);
//    cv::imshow("resize_img", resize_img);
    cv::waitKey(0);
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
