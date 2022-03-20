#include "stdafx.h"



SemiGlobalMatching::SemiGlobalMatching(){
    
}


SemiGlobalMatching::~SemiGlobalMatching()
{
    if (census_left_ != nullptr) {
        delete[] census_left_;
        census_left_ = nullptr;
    }
    if (census_right_ != nullptr) {
        delete[] census_right_;
        census_right_ = nullptr;
    }
    if (cost_init_ != nullptr) {
        delete[] cost_init_;
        cost_init_ = nullptr;
    }
    if (cost_aggr_ != nullptr) {
        delete[] cost_aggr_;
        cost_aggr_ = nullptr;
    }
    //各个方向代价聚合清空
    if (cost_aggr_1_ != nullptr) {
        delete[] cost_aggr_1_;
        cost_aggr_1_ = nullptr;
    }
    if (cost_aggr_2_ != nullptr) {
        delete[] cost_aggr_2_;
        cost_aggr_2_ = nullptr;
    }
     if (cost_aggr_3_ != nullptr) {
        delete[] cost_aggr_3_;
        cost_aggr_3_ = nullptr;
    }
     if (cost_aggr_4_ != nullptr) {
        delete[] cost_aggr_4_;
        cost_aggr_4_ = nullptr;
    }
     if (cost_aggr_5_ != nullptr) {
        delete[] cost_aggr_5_;
        cost_aggr_5_ = nullptr;
    }
     if (cost_aggr_6_ != nullptr) {
        delete[] cost_aggr_6_;
        cost_aggr_6_ = nullptr;
    }
     if (cost_aggr_7_ != nullptr) {
        delete[] cost_aggr_7_;
        cost_aggr_7_ = nullptr;
    }
     if (cost_aggr_8_ != nullptr) {
        delete[] cost_aggr_8_;
        cost_aggr_8_ = nullptr;
    }
    if(disp_left_ != nullptr) {
        delete[] disp_left_;
        disp_left_ = nullptr;
    }
    is_initialized_ = false;
}


bool SemiGlobalMatching::Initialize(const uint32& width, const uint32& height, const SGMOption& option){

    // ··· 赋值
    
	// 影像尺寸
    width_ = width;
    height_ = height;
    // SGM参数
    option_ = option;

    if(width == 0 || height == 0) {
        return false;
    }

    //··· 开辟内存空间

    // census值（左右影像）
    census_left_ = new uint32[width * height]();
    census_right_ = new uint32[width * height]();

    // 匹配代价（初始/聚合）
    const sint32 disp_range = option.max_disparity - option.min_disparity;
    if (disp_range <= 0) {
        return false;
    }
    cost_init_ = new uint8[width * height * disp_range]();

    //代价聚合各个方向
    cost_aggr_1_ = new uint8[width * height * disp_range]();
    cost_aggr_2_ = new uint8[width * height * disp_range]();
    cost_aggr_3_ = new uint8[width * height * disp_range]();
    cost_aggr_4_ = new uint8[width * height * disp_range]();
    cost_aggr_5_ = new uint8[width * height * disp_range]();
    cost_aggr_6_ = new uint8[width * height * disp_range]();
    cost_aggr_7_ = new uint8[width * height * disp_range]();
    cost_aggr_8_ = new uint8[width * height * disp_range]();

    //代价聚合值
    cost_aggr_ = new uint16[width * height * disp_range]();

    // 视差图
    disp_left_ = new float32[width * height]();

    is_initialized_ = census_left_ && census_right_ && cost_init_ && cost_aggr_ && disp_left_;

    return is_initialized_;
}

bool SemiGlobalMatching::Match(const uint8* img_left, const uint8* img_right, float32* disp_left)
{
    if(!is_initialized_) {
        return false;
    }
    if (img_left == nullptr || img_right == nullptr) {
        return false;
    }

    img_left_ =  img_left;
    img_right_ = img_right;


    // census变换
    CensusTransform();

    // 代价计算
    ComputeCost();

    // 代价聚合
    CostAggregation();
    
    // 视差计算
    ComputeDisparity();

    // 输出视差图
    memcpy(disp_left, disp_left_, width_ * height_ * sizeof(float32));

	return true;
}


bool SemiGlobalMatching::Reset(const uint32& width, const uint32& height, const SGMOption& option)
{
    // 释放内存
    if (census_left_ != nullptr) {
        delete[] census_left_;
        census_left_ = nullptr;
    }
    if (census_right_ != nullptr) {
        delete[] census_right_;
        census_right_ = nullptr;
    }
    if (cost_init_ != nullptr) {
        delete[] cost_init_;
        cost_init_ = nullptr;
    }
    if (cost_aggr_ != nullptr) {
        delete[] cost_aggr_;
        cost_aggr_ = nullptr;
    }
    if (disp_left_ != nullptr) {
        delete[] disp_left_;
        disp_left_ = nullptr;
    }

    // 重置初始化标记
    is_initialized_ = false;

	// 初始化
    return Initialize(width, height, option);
}

void SemiGlobalMatching::CensusTransform() const
{
	// 左右影像census变换
    sgm_util::census_transform_5x5(img_left_, census_left_, width_, height_);

    sgm_util::census_transform_5x5(img_right_, census_right_, width_, height_);
    
}


void SemiGlobalMatching::ComputeCost() const
{
    const sint32& min_disparity = option_.min_disparity;
    const sint32& max_disparity = option_.max_disparity;
    
    //视差范围
    const uint32 disp_range = max_disparity - min_disparity;

	// 计算代价（基于Hamming距离）
    for (sint32 i = 0; i < height_; i++) {
        for (sint32 j = 0; j < width_; j++) {

            // 左影像census值
            const uint32 census_val_l = census_left_[i * width_ + j];

            // 逐视差计算代价值
        	for (sint32 d = min_disparity; d < max_disparity; d++) {
                uint8& cost = cost_init_[i * width_ * disp_range + j * disp_range + (d - min_disparity)];
                if (j - d < 0 || j - d >= width_) {
                    cost = UINT8_MAX/2;     //这些视差在右视图上找不到对应像素，没法计算代价，只好给一个固定的较大值，表明这些视差代价很大。
                    continue;
                }
                // 右影像对应像点的census值
                const uint32 census_val_r = census_right_[i * width_ + j - d];
                
        		// 计算匹配代价
                cost = sgm_util::hamDist32(census_val_l, census_val_r);
            }
        }
    }
}

void SemiGlobalMatching::CostAggregation() const
{
    // 路径聚合
    // 1、左->右/右->左
    // 2、上->下/下->上
    // 3、左上->右下/右下->左上
    // 4、右上->左上/左下->右上
    //
    // ↘ ↓ ↙   5  3  7
    // →   ←   1     2
    // ↗ ↑ ↖   8  4  6
    //
    const auto& min_disparity = option_.min_disparity;
    const auto& max_disparity = option_.max_disparity;
    assert(max_disparity > min_disparity);

    const sint32 size = width_ * height_ * (max_disparity - min_disparity);
    if(size <= 0) {
        return;
    }

    const auto& P1 = option_.p1;
    const auto& P2_Int = option_.p2_int;

    // 左右聚合
    sgm_util::CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_1_, true);
    sgm_util::CostAggregateLeftRight(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_2_, false);
    
    // 上下聚合
    sgm_util::CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_3_, true);
    sgm_util::CostAggregateUpDown(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_4_, false);

    // 对角线聚合1
    sgm_util::CostAggregateDagonal_1(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_5_, true);
    sgm_util::CostAggregateDagonal_1(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_6_, false);

    // 对角线聚合2
    sgm_util::CostAggregateDagonal_2(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_5_, true);
    sgm_util::CostAggregateDagonal_2(img_left_, width_, height_, min_disparity, max_disparity, P1, P2_Int, cost_init_, cost_aggr_6_, false);
    
    // 把4/8个方向加起来
    for(sint32 i =0;i<size;i++) {
    	cost_aggr_[i] = cost_aggr_1_[i] + cost_aggr_2_[i] + cost_aggr_3_[i] + cost_aggr_4_[i];
    	if (option_.num_paths == 8) {
            cost_aggr_[i] += cost_aggr_5_[i] + cost_aggr_6_[i] + cost_aggr_7_[i] + cost_aggr_8_[i];
        }
    }
}


void SemiGlobalMatching::ComputeDisparity() const
{
	// 最小最大视差
    const sint32& min_disparity = option_.min_disparity;
    const sint32& max_disparity = option_.max_disparity;

    //视差范围
    const uint32 disp_range = max_disparity - min_disparity;

    // 代价聚合
    auto cost_ptr = cost_aggr_;

    // 逐像素计算最优视差
    for (sint32 i = 0; i < height_; i++) {
        for (sint32 j = 0; j < width_; j++) {
            
            uint16 min_cost = UINT16_MAX;
            uint16 max_cost = 0;
            sint32 best_disparity = 0;

            // 遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
            for (sint32 d = min_disparity; d < max_disparity; d++) {
                const sint32 d_idx = d - min_disparity;
                const auto& cost = cost_ptr[i * width_ * disp_range + j * disp_range + d_idx];
                if(min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
                max_cost = std::max(max_cost, static_cast<uint16>(cost));
            }

            // 最小代价值对应的视差值即为像素的最优视差
            if (max_cost != min_cost) {
                disp_left_[i * width_ + j] = static_cast<float>(best_disparity);
            }
            else {
            	// 如果所有视差下的代价值都一样，则该像素无效
                disp_left_[i * width_ + j] = Invalid_Float;
            }
        }
    }
}
