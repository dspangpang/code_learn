#pragma once

#ifndef SAFE_DELETE
#define SAFE_DELETE(P) {if(P) delete[](P);(P)=nullptr;}
#endif

/** 
 * \brief 储存3D坐标的结构体
 * \param x x坐标
 * \param y y坐标
 * \param z z坐标
 */
struct XYZ
{
    double x;        //x坐标
    double y;        //y坐标
    double z;        //z坐标
};

/** 
 * \brief 立方体网格的信息
 * \param p 顶点坐标
 * \param n 法向量
 * \param val 顶点对应的灰度值
 */
struct GRIDCELL
{
    XYZ p[8];        //顶点坐标
    XYZ n[8];        //法向量
    double val[8];   //顶点对应的灰度值
};

/** 
 * \brief 三角形面的信息
 * \param p 顶点坐标
 * \param c 几何中心
 * \param n 法向量
 */
struct TRIANGLE
{
    XYZ p[3];        //顶点坐标
    XYZ c;           //几何中心
    XYZ n[3];        ///法向量
};

/** 
 * \brief 目标点云的范围
 * \param Max_x 最大x轴坐标
 * \param Max_y 最大y轴坐标
 * \param Max_z 最大z轴坐标
 * \param GridSize 每个网格的大小
 */
struct MC_Felid{

   double Max_x;        //最大x轴坐标
   double Max_y;        //最大y轴坐标
   double Max_z;        //最大z轴坐标
   double GridSize;     //每个网格的大小

   double isovaulel;
};
