#include<iostream>

#include<opencv2/opencv.hpp>

int main(int argc, char * argv[]){


    cv::Mat src = cv::imread("./love.jpg");         //B G R 为读入顺序
    cv::Mat m1 = src.clone();

    cv::Mat dst_add = src.clone();
    cv::Mat dst_div = src.clone();
    cv::Mat dst_sub = src.clone();
    cv::Mat dst_mul = src.clone();


    if(src.empty()){
        std::cout << "loading picture failure \n";
        return -1;
    }
    m1 = cv::Scalar(2, 2, 2); 
    cv::add(src, m1, dst_add);
    cv::subtract(src, m1, dst_sub);
    cv::multiply(src, m1, dst_mul);
    cv::divide(src, m1, dst_div);

    cv::imshow("+", dst_add);
    cv::imshow("-", dst_sub);
    cv::imshow("*", dst_mul);
    cv::imshow("/", dst_div);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}