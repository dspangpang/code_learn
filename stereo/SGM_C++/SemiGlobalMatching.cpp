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

void census_transform_5x5(const uint8* source, uint32* census, const sint32& width,const sint32& height)
{
	if (source == nullptr || census == nullptr || width <= 5u || height <= 5u) {
		return;
	}

	// 逐像素计算census值
	for (sint32 i = 2; i < height - 2; i++) {
		for (sint32 j = 2; j < width - 2; j++) {
			
			// 中心像素值
			const uint8 gray_center = source[i * width + j];
			
			// 遍历大小为5x5的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
			uint32 census_val = 0u;
			for (sint32 r = -2; r <= 2; r++) {
				for (sint32 c = -2; c <= 2; c++) {
					census_val <<= 1;
					const uint8 gray = source[(i + r) * width + j + c];
					if (gray < gray_center) {
						census_val += 1;
					}
				}
			}

			// 中心像素的census值
			census[i * width + j] = census_val;		
		}
	}
}

void SemiGlobalMatching::CensusTransform() const
{
	// 左右影像census变换
    census_transform_5x5(img_left_, census_left_, width_, height_);

    census_transform_5x5(img_right_, census_right_, width_, height_);
    
}


// unsigned to 32bit
//计算hamming距离
uint32 hamDist32(const uint32& x, const uint32& y)
{
	uint32 dist = 0, val = x ^ y;

	// Count the number of set bits
	while (val)
	{
		++dist;
		val &= val - 1;
	}

	return dist;
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
                cost = hamDist32(census_val_l, census_val_r);
            }
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

    // 未实现聚合步骤，暂用初始代价值来代替
    auto cost_ptr = cost_init_;

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
