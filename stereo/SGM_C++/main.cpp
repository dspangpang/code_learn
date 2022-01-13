#include"stdafx.h"

/**
 * \brief 
 * \param argv 3
 * \param argc argc[1]:左影像路径 argc[2]: 右影像路径 argc[3]: 视差图路径
 * \return 
 */
int main(int argv,char** argc)
{

    // ··· 读取影像
   
    cv::Mat img_left = cv::imread("./im2.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread("./im6.png", cv::IMREAD_GRAYSCALE);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "读取影像失败！" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "左右影像尺寸不一致！" << std::endl;
        return -1;
    }

    // ··· SGM匹配
    const uint32 width = static_cast<uint32>(img_left.cols);
    const uint32 height = static_cast<uint32>(img_right.rows);

    SemiGlobalMatching::SGMOption sgm_option;
    sgm_option.num_paths = 8;
    sgm_option.min_disparity = 0;
    sgm_option.max_disparity = 64;
    sgm_option.p1 = 10;
    sgm_option.p2_int = 150;

    SemiGlobalMatching sgm;

    // 初始化
    if(!sgm.Initialize(width, height, sgm_option)) {
        std::cout << "SGM初始化失败！" << std::endl;
        return -2;
    }

    // 匹配
    auto disparity = new float32[width * height]();
    if(!sgm.Match(img_left.data, img_right.data, disparity)) {
        std::cout << "SGM匹配失败！" << std::endl;
        return -2;
    }

    // 显示视差图
    cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);
    for (uint32 i=0;i<height;i++) {
	    for(uint32 j=0;j<width;j++) {
            const float32 disp = disparity[i * width + j];
            if (disp == Invalid_Float) {
                disp_mat.data[i * width + j] = 0;
            }
            else {
                disp_mat.data[i * width + j] = 2 * static_cast<uchar>(disp);
            }
	    }
    }

    delete[] disparity;
    disparity = nullptr;

    cv::imwrite("视差图.png", disp_mat);
    cv::imshow("视差图", disp_mat);
    cv::waitKey(0);
    cv::destroyAllWindows();


    
	return 0;
}
