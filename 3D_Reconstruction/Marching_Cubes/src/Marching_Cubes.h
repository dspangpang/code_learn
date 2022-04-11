#pragma once

#include <iostream>
#include "Cubes_table.h"
#include <Marching_Cubes_types.h>
#include <math.h>

#define ABS(x) ((x < 0) ? (-x) : (x))

class Marching_Cubes
{

public:

   Marching_Cubes();
   ~Marching_Cubes();

   /** \brief 获取生产的网格的数量
    */
   void GetGridNum(MC_Felid& a);

   /** \brief 空间初始化
    */
   void Initialize();
   
   /** \brief 空间释放
    */
   void Release();

   /** \brief 计算出建立小正方体内的三角形面的位置参数
    * \param grid 小立方体的参数信息
    * \param isolevel 给定的等值面
    * \param triangles 建立的三角形的位置参数
    * \return 在该小正方体内建立的三角形的个数
    */
   int Polygonise(GRIDCELL grid,double isolevel,TRIANGLE *triangles);

   /** \brief 线性插值计算曲面与小正方体边的交点位置
    * \param isolevel 给定的等值面
    * \param p1 小正方体交线上的点1
    * \param p2 小正方体交线上的点2
    * \param valp1 小正方体交线上的点1对应的值
    * \param valp2 小正方体交线上的点2对应的值
    * \return 计算后得到的交点坐标
    */
   XYZ VertexInterp(double isolevel,XYZ p1,XYZ p2,double valp1,double valp2);

   /** \brief 所有小正方体的三角形截面计算
    */
   void MarchingCubes_Compute();

private:

   /** \brief X方向栅格数目 */
   int NX;

   /** \brief Y方向栅格数目 */
   int NY;

   /** \brief Z方向栅格数目 */
   int NZ;

   /** \brief 所有三角形数据存储 */
   TRIANGLE* triangle;

   /** \brief 所有三角形数据存储 */
   GRIDCELL* gridcell;

};

