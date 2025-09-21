
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int, char**){
    std::cout << "Hello, from homework2.1!\n";
    // 读取图片
    cv::Mat img = cv::imread("../resources/test_image_2.png");
    if (img.empty()) {
        std::cerr << "无法加载图片！" << std::endl;
        return 1;
    }

    // 对原图进行中值滤波
    cv::Mat img_median;
    cv::medianBlur(img, img_median, 5);

    // 基于梯度的亮度边缘检测
    cv::Mat gray;
    cv::cvtColor(img_median, gray, cv::COLOR_BGR2GRAY);
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    cv::Mat grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);
    cv::Mat grad_norm;
    cv::normalize(grad_mag, grad_norm, 0, 255, cv::NORM_MINMAX);
    grad_norm.convertTo(grad_norm, CV_8U);

    // 仅保留亮度边缘检测结果中最明显的5个轮廓，并用红色矩形框出外接矩形
    cv::Mat bright_bin;
    cv::threshold(grad_norm, bright_bin, 50, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bright_bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
        return cv::contourArea(a) > cv::contourArea(b);
    });
    cv::Mat contour_vis = img_median.clone();
    int keep_n = std::min(5, (int)contours.size());
    // 去除面积最大的三个，仅绘制第4、5大轮廓
    for (int i = 3; i < keep_n; ++i) {
        cv::Rect bbox = cv::boundingRect(contours[i]);
        cv::rectangle(contour_vis, bbox, cv::Scalar(0,0,255), 2);
    }

    // 保存最终结果
    cv::imwrite("../resources/version2.png", contour_vis);
    // 只显示最终结果
    cv::imshow("Top 5 Brightness Edge BoundingBox", contour_vis);
    cv::waitKey(0);
    return 0;

    cv::waitKey(0);
    return 0;
}

