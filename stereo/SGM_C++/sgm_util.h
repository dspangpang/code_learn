#pragma once

#include"stdafx.h"



namespace sgm_util
{

/**
 * \brief census变换
 * \param source	输入，影像数据
 * \param census	输出，census值数组
 * \param width		输入，影像宽
 * \param height	输入，影像高
 */
void census_transform_5x5(const uint8* source, uint32* census, const sint32& width, const sint32& height);

/**
 * \brief haming距离计算
 * \param x	输入数据1
 * \param y	输入数据1
 * \return haming距离的值
 */
uint32 hamDist32(const uint32& x, const uint32& y);

/**
 * \brief 左右方向路径聚合
 * \param img_data 输入图像地址
 * \param width	图像宽
 * \param height 图像高
 * \param min_disparity 最小视差	
 * \param max_disparity 最大视差
 * \param p1 惩罚系数P1
 * \param p2_init 惩罚系数P2
 * \param cost_init 初始代价
 * \param cost_aggr 代价聚合
 * \param is_forward 控制聚合方向 1/0
 */
void CostAggregateLeftRight(const uint8* img_data, const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity,
	const sint32& p1, const sint32& p2_init, const uint8* cost_init, uint8* cost_aggr, bool is_forward);

/**
 * \brief 上下方向路径聚合
 * \param img_data 输入图像地址
 * \param width	图像宽
 * \param height 图像高
 * \param min_disparity 最小视差	
 * \param max_disparity 最大视差
 * \param p1 惩罚系数P1
 * \param p2_init 惩罚系数P2
 * \param cost_init 初始代价
 * \param cost_aggr 代价聚合
 * \param is_forward 控制聚合方向 1/0
 */
void CostAggregateUpDown(const uint8* img_data, const sint32& width, const sint32& height,const sint32& min_disparity, const sint32& max_disparity, 
	const sint32& p1, const sint32& p2_init,const uint8* cost_init, uint8* cost_aggr, bool is_forward);

}