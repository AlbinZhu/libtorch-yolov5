//
// Created by swing on 8/1/20.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

std::vector<float> imagePadding(const cv::Mat& src, cv::Mat& dst, const cv::Size& out_size) {
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);

    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h +1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};

    return pad_info;
}

torch::Tensor GetBoundingBoxIoU(const torch::Tensor& box1, const torch::Tensor& box2) {
    const torch::Tensor& b1_x1 = box1.select(1, 0);
    const torch::Tensor& b1_y1 = box1.select(1, 1);
    const torch::Tensor& b1_x2 = box1.select(1, 2);
    const torch::Tensor& b1_y2 = box1.select(1, 3);

    const torch::Tensor& b2_x1 = box2.select(1, 0);
    const torch::Tensor& b2_y1 = box2.select(1, 1);
    const torch::Tensor& b2_x2 = box2.select(1, 2);
    const torch::Tensor& b2_y2 = box2.select(1, 3);

    // 交集坐标
    torch::Tensor inter_rect_x1 = torch::max(b1_x1, b2_x1);
    torch::Tensor inter_rect_y1 = torch::max(b1_y1, b2_y1);
    torch::Tensor inter_rect_x2 = torch::min(b1_x2, b2_x2);
    torch::Tensor inter_rect_y2 = torch::min(b1_y2, b2_y2);

    // 交集面积
    torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1 + 1, torch::zeros(inter_rect_x2.sizes())) *
            torch::max(inter_rect_y2 - inter_rect_y1 + 1, torch::zeros(inter_rect_x2.sizes()));

    // 并集面积
    torch::Tensor b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1);
    torch::Tensor b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1);

    // iou
    torch::Tensor iou = inter_area / (b1_area + b2_area - inter_area);
    return iou;

}


torch::Tensor PostProcessing(const torch::Tensor& detections, float conf_thres, float iou_thres) {
    constexpr int item_attr_size = 5;
    int batch_size = detections.size(0);
    auto num_classes = detections.size(2) - item_attr_size;
    auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

    detections.slice(2, item_attr_size, item_attr_size + num_classes) *=
            detections.select(2, 4).unsqueeze(2);

    torch::Tensor box = torch::zeros(detections.sizes(), detections.options());
    box.select(2, 0) = detections.select(2, 0) - detections.select(2, 2).div(2);
    box.select(2, 1) = detections.select(2, 1) - detections.select(2, 3).div(2);
    box.select(2, 2) = detections.select(2, 0) + detections.select(2, 2).div(2);
    box.select(2, 3) = detections.select(2, 1) + detections.select(2, 3).div(2);
    detections.slice(2, 0, 4) = box.slice(2, 0, 4);

    bool is_initialized = false;
    torch::Tensor output = torch::zeros({1, 7});

    for (int i = 0; i < batch_size; ++i) {
        auto det = torch::masked_select(detections[i], conf_mask[i]).view({-1, num_classes + item_attr_size});

        if (det.size(1) == 0) {
            continue;
        }

        std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);
        auto max_conf_score = std::get<0>(max_classes);
        auto max_conf_index = std::get<1>(max_classes);

        max_conf_score = max_conf_score.to(torch::kFloat32).unsqueeze(1);
        max_conf_index = max_conf_index.to(torch::kFloat32).unsqueeze(1);

        det = torch::cat({det.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

        std::vector<torch::Tensor> img_classes;

        auto len = det.size(0);
        for (int j = 0; j < len; ++j) {
            bool found = false;
            for (const auto& cls : img_classes) {
                auto ret = (det[j][5] == cls);
                if (torch::nonzero(ret).size(0) > 0) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                img_classes.emplace_back(det[j][5]);
            }
        }

        for (const auto& cls : img_classes) {
            auto cls_mask = det * (det.select(1, 5) == cls).to(torch::kFloat32).unsqueeze(1);
            auto cls_mask_index = torch::nonzero(cls_mask.select(1,0)).squeeze();

            auto bbox_by_class = det.index_select(0, cls_mask_index).view({-1, 6});

            std::tuple<torch::Tensor, torch::Tensor> sort_ret = torch::sort(bbox_by_class.select(1, 4), -1, true);
            auto conf_sort_index = std::get<1>(sort_ret);
            bbox_by_class = bbox_by_class.index_select(0, conf_sort_index).cpu();
            int num_by_class = bbox_by_class.size(0);

            // NMS
            for (int j = 0; j < num_by_class - 1; ++j) {
                auto iou = GetBoundingBoxIoU(bbox_by_class[j].unsqueeze(0), bbox_by_class.slice(0, j + 1, num_by_class));
                auto iou_mask = (iou < iou_thres).to(torch::kFloat32).unsqueeze(1);
                bbox_by_class.slice(0, j + 1, num_by_class) *= iou_mask;

                auto non_zero_index = torch::nonzero(bbox_by_class.select(1, 4)).squeeze();
                bbox_by_class = bbox_by_class.index_select(0, non_zero_index).view({-1, 6});

                num_by_class = bbox_by_class.size(0);
            }

            torch::Tensor batch_index = torch::zeros({bbox_by_class.size(0), 1}).fill_(i);

            if (!is_initialized) {
                output = torch::cat({batch_index, bbox_by_class}, 1);
                is_initialized = true;
            }
            else {
                auto out = torch::cat({batch_index, bbox_by_class}, 1);
                output = torch::cat({output, out}, 0);
            }

        }
    }
    return output;
}