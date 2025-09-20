#include <iostream>
#include <iostream>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main(int, char**){

    // 读取原始图片
    std::string img_path = "../resources/test_image_2.png";
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "无法加载图片: " << img_path << std::endl;
        return 1;
    }

    // 转为HSV图
    cv::Mat hsv_img;
    cv::cvtColor(img, hsv_img, cv::COLOR_BGR2HSV);

    // HSV分割#FFFBFF颜色
    cv::Mat bgr_pixel(1,1,CV_8UC3,cv::Scalar(255,251,255));
    cv::Mat hsv_pixel;
    cv::cvtColor(bgr_pixel, hsv_pixel, cv::COLOR_BGR2HSV);
    cv::Vec3b hsv_val = hsv_pixel.at<cv::Vec3b>(0,0);
    int h = hsv_val[0], s = hsv_val[1], v = hsv_val[2];
    cv::Scalar lower_hsv(std::max(h-5,0), std::max(s-30,0), std::max(v-30,0));
    cv::Scalar upper_hsv(std::min(h+5,180), std::min(s+30,255), std::min(v+30,255));
    cv::Mat mask_FFFBFF;
    cv::inRange(hsv_img, lower_hsv, upper_hsv, mask_FFFBFF);

    // 高斯滤波去噪
    cv::Mat mask_blur;
    cv::GaussianBlur(mask_FFFBFF, mask_blur, cv::Size(5,5), 0);

    // 轮廓识别
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask_blur, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 绘制轮廓和bounding box
    cv::Mat contour_img = img.clone();
    std::vector<cv::Rect> selected_boxes;
    for (size_t i = 0; i < contours.size(); ++i) {
        int label_num = static_cast<int>(i+1);
        if (label_num == 34 || label_num == 39) {
            cv::Rect bbox = cv::boundingRect(contours[i]);
            cv::rectangle(contour_img, bbox, cv::Scalar(0,0,255), 2);
            selected_boxes.push_back(bbox);
        }
    }

    // 连接两个矩形框形成一个大矩形框
    if (selected_boxes.size() == 2) {
        int x1 = std::min(selected_boxes[0].x, selected_boxes[1].x);
        int y1 = std::min(selected_boxes[0].y, selected_boxes[1].y);
        int x2 = std::max(selected_boxes[0].x + selected_boxes[0].width, selected_boxes[1].x + selected_boxes[1].width);
        int y2 = std::max(selected_boxes[0].y + selected_boxes[0].height, selected_boxes[1].y + selected_boxes[1].height);
        cv::Rect big_box(x1, y1, x2-x1, y2-y1);
        cv::Mat big_box_img = img.clone();
        cv::rectangle(big_box_img, big_box, cv::Scalar(0,0,255), 2);
        cv::imwrite("../output/armop_image_3.png", big_box_img);
        cv::imshow("armop_image_3.png", big_box_img);
    }
    cv::imwrite("../output/armor_detection_v2_result.jpg", contour_img);
    cv::imshow("armor_detection_v2_result.jpg", contour_img);

    cv::waitKey(0);
    return 0;
}

