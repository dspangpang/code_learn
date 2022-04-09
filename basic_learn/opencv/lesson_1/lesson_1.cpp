#include<iostream>
#include<string>

#include<opencv2/opencv.hpp>

int main(int argc, char * argv[]){

    cv::Mat img = cv::imread("./love.jpg");

    if(img.empty()){
        std::cout << "loading picture failure \n";
        return -1;
    }

    cv::namedWindow("src", cv::WINDOW_FREERATIO); //设置命名窗口出现的方式

    cv::imshow("src", img);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}