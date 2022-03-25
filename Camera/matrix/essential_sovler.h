#pragma once

#include "./eigen_defs.h"

namespace sv3d
{
    class EssentialSolver
    {
    public:
        EssentialSolver() = default;
        ~EssentialSolver() = default;

        enum SOLVE_TYPE{
            EIGHT_POINTS = 0
        };

        /**
		 * \brief 本质矩阵求解
		 * \param p1[in] 视图1上像素点齐次坐标
		 * \param p2[in] 视图2上像素点齐次坐标
		 * \param k1_mat[in] 相机1内参矩阵
		 * \param k2_mat[in] 相机2内参矩阵
		 * \param solver_type[in] 求解算法类型
		 */
		void Solve(const Mat3X p1, const Mat3X p2, const Mat3 k1_mat, const Mat3 k2_mat, const SOLVE_TYPE& solver_type);
		
        /**
		 * \brief 本质矩阵求解
		 * \param x1[in] 视图1上的归一化像素点齐次坐标(x = k_inv*p)
		 * \param x2[in] 视图2上的归一化像素点齐次坐标
		 * \param solver_type[in] 求解算法类型
		 */
		void Solve(const Mat3X x1, const Mat3X x2, const SOLVE_TYPE& solver_type);

		/**
		 * \brief 获取本质矩阵
		 * \return 本质矩阵
		 */
		Mat3 Value();
	private:

		/**
		 * \brief 八点法求解本质矩阵
		 * \param x1[in] 视图1上的归一化像素点齐次坐标(x = k_inv*p)
		 * \param x2[in] 视图2上的归一化像素点齐次坐标
		 */
		void Solve_EightPoints(const Mat3X x1, const Mat3X x2);

		//void Solve_FivePoints(const Mat3X x1, const Mat3X x2);

		/* 本质矩阵数据 */
		Mat3 data_;

    };
    
}