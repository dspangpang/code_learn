#pragma once

#include<eigen3/Eigen/Eigen>

namespace sv3d
{
    // 2d 向量 (double类型)
	using Vec2 = Eigen::Vector2d;
		
	// 2d 向量 (float类型)
	using Vec2f = Eigen::Vector2f;
	
	// 3d 向量 (double类型)
	using Vec3 = Eigen::Vector3d;

	// 3d 向量 (float类型)
	using Vec3f = Eigen::Vector3f;

	// 4d 向量 
	using Vec4 = Eigen::Vector4d;
	
	// 9d 向量
	using Vec9 = Eigen::Matrix<double, 9, 1>;
	
	// 3x3 矩阵 (double类型)
	using Mat3 = Eigen::Matrix<double, 3, 3>;
		
	// 3x4 矩阵 (double类型)
	using Mat34 = Eigen::Matrix<double, 3, 4>;
	
	// 3x3 矩阵 (double类型) 行优先的存储方式
	using RMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

	// 3x4 矩阵 (double类型) 行优先的存储方式
	using RMat34 = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

	// 4x4 矩阵 (double类型)
	using Mat4 = Eigen::Matrix<double, 4, 4>;
	
	// 2xN 矩阵 (double类型) 动态矩阵
	using Mat2X = Eigen::Matrix<double, 2, Eigen::Dynamic>;
		
	// 3xN 矩阵 (double类型) 动态矩阵
	using Mat3X = Eigen::Matrix<double, 3, Eigen::Dynamic>;

	// Nx9 矩阵 (double类型) 动态矩阵
	using MatX9 = Eigen::Matrix<double, Eigen::Dynamic, 9>;

	// Nx9 矩阵 (double类型) 动态矩阵 行优先的存储方式
	using RMatX9 = Eigen::Matrix<double, Eigen::Dynamic, 9, Eigen::RowMajor>;

	// NxM 矩阵 (double类型) 动态矩阵
	using MatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

	// NxM 矩阵 (double类型) 动态矩阵 行优先的存储方式
	using RMatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

} // namespace name

/*
1.如果要和其他库合作开发，为了转化方便，可以选择同样的存储方式。
2.应用中涉及大量行遍历操作，应该选择行优先，寻址更快。反之亦然。
3.默认是列优先，而且大多库都是按照这个顺序的，默认的不失为较好的。*/