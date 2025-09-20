#include <iostream>
#include <opencv2/opencv.hpp>

int main(){
    // 读取彩色图片
    cv::Mat img = cv::imread("resources/test_image.png");
    if(img.empty()) {
        std::cerr << "无法加载图片 resources/test_image.png" << std::endl;
        return 1;
    }
    // 图像裁剪为左上角1/4
    int crop_w = img.cols / 2;
    int crop_h = img.rows / 2;
    cv::Rect roi(0, 0, crop_w, crop_h);
    cv::Mat img_cropped = img(roi);
    cv::imwrite("../output/cropped_top_left.png", img_cropped);
    cv::imshow("Cropped Top-Left 1/4", img_cropped);
    cv::waitKey(0);

    // 图像旋转35度
    cv::Point2f center(img.cols/2.0F, img.rows/2.0F);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, 35, 1.0);
    cv::Mat img_rotated;
    cv::warpAffine(img, img_rotated, rot_mat, img.size());
    cv::imwrite("../output/rotated.png", img_rotated);
    cv::imshow("Rotated Image", img_rotated);
    cv::waitKey(0);

    // 转为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // 显示灰度图
    cv::imwrite("../output/gray.png", gray);
    cv::imshow("Gray Image", gray);
    cv::waitKey(0);
    // 转为HSV图
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    // 显示HSV图（注意：直接显示HSV效果不直观，通常用于后续处理）
    cv::imwrite("../output/hsv.png", hsv);
    cv::imshow("HSV Image", hsv);
    cv::waitKey(0);

    // 对HSV图像应用均值滤波
    cv::Mat hsv_blur;
    cv::blur(hsv, hsv_blur, cv::Size(5, 5));
    cv::imwrite("../output/hsv_mean_blur.png", hsv_blur);
    cv::imshow("HSV Mean Blur", hsv_blur);
    cv::waitKey(0);

    // 对均值滤波后的图像应用高斯滤波
    cv::Mat hsv_gauss;
    cv::GaussianBlur(hsv_blur, hsv_gauss, cv::Size(5, 5), 0);
    cv::imwrite("../output/hsv_gaussian_blur.png", hsv_gauss);
    cv::imshow("HSV Gaussian Blur", hsv_gauss);
    cv::waitKey(0);

    // HSV方法提取红色区域
    cv::Mat red_mask1, red_mask2, red_mask;
    cv::inRange(hsv_gauss, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), red_mask1);
    cv::inRange(hsv_gauss, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), red_mask2);
    cv::bitwise_or(red_mask1, red_mask2, red_mask);
    cv::imwrite("../output/red_area_mask.png", red_mask);
    cv::imshow("Red Area Mask", red_mask);
    cv::waitKey(0);

    // 寻找红色区域外轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 在原图上绘制外轮廓和bounding box
    cv::Mat result = img.clone();
    // 绘制任意圆形
    cv::circle(result, cv::Point(100, 100), 50, cv::Scalar(0, 255, 255), 3);
    // 绘制任意方形
    cv::rectangle(result, cv::Rect(200, 50, 80, 80), cv::Scalar(255, 255, 0), 3);
    // 绘制任意文字
    cv::putText(result, "OpenCV Demo", cv::Point(50, 250), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255,0,255), 2);

    // 绘制红色外轮廓和bounding box
    for(size_t i = 0; i < contours.size(); ++i) {
        cv::drawContours(result, std::vector<std::vector<cv::Point>>{contours[i]}, -1, cv::Scalar(0,255,0), 2);
        cv::Rect bbox = cv::boundingRect(contours[i]);
        cv::rectangle(result, bbox, cv::Scalar(255,0,0), 2);
        double area = cv::contourArea(contours[i]);
        std::cout << "轮廓 " << i << " 的面积: " << area << std::endl;
    }
    cv::imwrite("../output/draw_shapes_red_features.png", result);
    cv::imshow("Draw Shapes & Red Features", result);
    cv::waitKey(0);
    // --------- 高亮区域处理流程 ---------
    // 1. 提取高亮区域（假设高亮为亮度高的区域，使用HSV的V通道阈值）
    cv::Mat highlight_mask;
    cv::inRange(hsv_gauss, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), highlight_mask);
    cv::imwrite("../output/highlight_mask.png", highlight_mask);
    cv::imshow("Highlight Mask", highlight_mask);
    cv::waitKey(0);

    // 2. 灰度化（高亮掩码本身已是单通道，可直接处理）
    cv::Mat highlight_gray = highlight_mask.clone();
    cv::imwrite("../output/highlight_gray.png", highlight_gray);
    cv::imshow("Highlight Gray", highlight_gray);
    cv::waitKey(0);

    // 3. 二值化
    cv::Mat highlight_bin;
    cv::threshold(highlight_gray, highlight_bin, 128, 255, cv::THRESH_BINARY);
    cv::imwrite("../output/highlight_binary.png", highlight_bin);
    cv::imshow("Highlight Binary", highlight_bin);
    cv::waitKey(0);

    // 4. 膨胀
    cv::Mat highlight_dilate;
    cv::dilate(highlight_bin, highlight_dilate, cv::Mat(), cv::Point(-1,-1), 2);
    cv::imwrite("../output/highlight_dilate.png", highlight_dilate);
    cv::imshow("Highlight Dilate", highlight_dilate);
    cv::waitKey(0);

    // 5. 腐蚀
    cv::Mat highlight_erode;
    cv::erode(highlight_dilate, highlight_erode, cv::Mat(), cv::Point(-1,-1), 2);
    cv::imwrite("../output/highlight_erode.png", highlight_erode);
    cv::imshow("Highlight Erode", highlight_erode);
    cv::waitKey(0);

    // 6. 漫水处理（以图像中心为种子点，填充高亮区域）
    cv::Mat flood = highlight_erode.clone();
    cv::Mat flood_mask = cv::Mat::zeros(flood.rows+2, flood.cols+2, CV_8UC1);
    cv::floodFill(flood, flood_mask, cv::Point(flood.cols/2, flood.rows/2), 128, 0, 0, 0, 4);
    cv::imwrite("../output/flood_fill.png", flood);
    cv::imshow("Flood Fill", flood);
    cv::waitKey(0);
    return 0;
}
