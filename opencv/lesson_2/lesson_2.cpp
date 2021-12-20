#include<iostream>

#include<opencv2/opencv.hpp>

int main(){

    cv::Mat src = cv::imread("./love.jpg");         //B G R 为读入顺序
    cv::Mat hsv;
    cv::Mat grey;

    if(src.empty()){
        std::cout << "loading picture failure \n";
        return -1;
    }

    cv::cvtColor(src, hsv, cv::COLOR_RGB2HSV);
    cv::cvtColor(src, grey, cv::COLOR_RGBA2GRAY);

    cv::imwrite("./hsv.jpg", hsv);   // H 0～180， S 0～255， V 0～255（亮度）
    cv::imwrite("./grey.jpg", grey);

    cv::namedWindow("src", cv::WINDOW_FREERATIO);
    cv::namedWindow("hsv", cv::WINDOW_FREERATIO);
    cv::namedWindow("grey", cv::WINDOW_FREERATIO);

    cv::imshow("src", src);      //仅支持8位图像或者浮点类型   
    cv::imshow("hsv", hsv);
    cv::imshow("grey", grey);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}