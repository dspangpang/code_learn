#include<iostream>

#include<opencv2/opencv.hpp>

int main(){

    cv::Mat src = cv::imread("./love.jpg");         //B G R 为读入顺序
    cv::Mat m1;
    cv::Mat m2;


    if(src.empty()){
        std::cout << "loading picture failure \n";
        return -1;
    }

    //图像的复制(都会把图片的数据空间也复制一份)
    m1 = src.clone();   
    src.copyTo(m2);

    //图像指针转移(两个对象指向同一个数据空间)对其中一个对象进行修改会影响所有的对象
    cv::Mat src_1 = src;


    //创建空白图像
    cv::Mat m3 = cv::Mat::zeros(cv::Size(512, 512), CV_8UC3);

    cv::Mat m4 = cv::Mat::ones(cv::Size(8, 8), CV_8UC3);   //只能让一个通道变为1
    
    m3 = cv::Scalar(4, 56, 251);    //给各个通道的数据赋值


    cv::imshow("color", m3);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}