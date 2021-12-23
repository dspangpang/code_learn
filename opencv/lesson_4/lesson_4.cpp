#include<iostream>

#include<opencv2/opencv.hpp>



int main(int argc, char * argv[]){

    cv::Mat src = cv::imread("./love.jpg");         //B G R 为读入顺序
    cv::Mat m1 = src.clone();
    cv::Mat m2 = src.clone();


    if(src.empty()){
        std::cout << "loading picture failure \n";
        return -1;
    }
    
    
    for(int i = 0; i < src.rows; i++){      //索引方式操作
        for(int j = 0; j < src.cols; j++){
            cv::Vec3b bgr = src.at<cv::Vec3b>(i, j);
            src.at<cv::Vec3b>(i, j)[0] = 255 - bgr[0];
            src.at<cv::Vec3b>(i, j)[1] = 255 - bgr[1];
            src.at<cv::Vec3b>(i, j)[2] = 255 - bgr[2];
        }
    }

    for(int i = 0; i < m2.rows; i++){       //指针方式操作
        uchar * current_row = m2.ptr<uchar>(i);
        for(int j = 0; j < m2.cols; j++){
            *current_row++ = 255 - *current_row; 
            *current_row++ = 255 - *current_row; 
            *current_row++ = 255 - *current_row; 
        }
    }

    cv::imshow("原图", m1);
    cv::imshow("反色1", src);
    cv::imshow("反色2", m2);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;  
}