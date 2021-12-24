#include<iostream>
#include<string>

#include<opencv2/opencv.hpp>

#include<cuda_runtime.h>
#include<cuda.h>
#include<cuda_runtime_api.h>

__global__ void rgb_to_grey(const uchar3 * const d_image_rgb, uchar * const d_image_grey, int pix){
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < pix){
        const uchar B = d_image_rgb[tid].x;
        const uchar G = d_image_rgb[tid].x;
        const uchar R = d_image_rgb[tid].x;

        d_image_grey[tid] = uchar(.299f * R + .587f * G + .114f * B);
    }
}


void global_rgb2grey(cv::Mat image_rgb, cv::Mat image_grey){

    
    int rows = image_rgb.rows;
    int cols = image_rgb.cols;
    int pix  = rows * cols;
    const int MaxThreadsPerBlock = 1024;
    const int BlocksperGrid = (pix + MaxThreadsPerBlock - 1) / MaxThreadsPerBlock;

    uchar3 *h_image_rgb;
    uchar  *h_image_grey;

    uchar3 *d_image_rgb;
    uchar *d_image_grey;

    h_image_grey = (uchar *)image_grey.ptr<uchar>(0);
    h_image_rgb  = (uchar3 *) image_rgb.ptr<uchar>(0);

    cudaMalloc(&d_image_rgb, sizeof(uchar3) * pix);
    cudaMalloc(&d_image_grey, sizeof(uchar) * pix);

    cudaMemcpy(d_image_rgb, h_image_rgb, sizeof(uchar3) * pix, cudaMemcpyHostToDevice);

    rgb_to_grey<<<BlocksperGrid, MaxThreadsPerBlock>>>(d_image_rgb, d_image_grey, pix);

    cudaMemcpy(h_image_grey, d_image_grey, sizeof(uchar) * pix, cudaMemcpyDeviceToHost);

    cudaFree(d_image_rgb);
    cudaFree(d_image_grey);

}



int main (int argc, char * argv[]){

    cv::Mat imgae_rgb = cv::imread("./love.jpg");
    cv::Mat image_grey = cv::Mat::zeros(cv::Size(imgae_rgb.cols, imgae_rgb.rows), CV_8UC1);

    // image_grey.create(imgae_rgba.rows, imgae_rgba.cols, CV_8UC1);  另一种Mat创建方法

    global_rgb2grey(imgae_rgb, image_grey);

    cudaDeviceSynchronize();

    cv::imshow("grey",image_grey); 
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;

}