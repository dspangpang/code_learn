#pragma once

#include <iostream>
#include "Cubes_table.h"

#define ABS(x) ((x < 0) ? (-x) : (x))

class Marching_Cubes
{

private:

/** \brief 储存3D坐标的结构体
 ** \param x x坐标
 ** \param y y坐标
 ** \param z z坐标
*/
struct XYZ
{
    double x;
    double y;
    double z; 
};

/** \brief 立方体网格的信息
 ** \param p 顶点坐标
 ** \param n 法向量
 ** \param val 顶点对应的灰度值
*/
struct GRIDCELL
{
    XYZ p[8];
    XYZ n[8];
    double val[8];     
};

/** \brief 三角形面的信息
 ** \param p 定点坐标
 ** \param c 几何中心
 ** \param n 法向量
*/
struct TRIANGLE
{
    XYZ p[3];
    XYZ c;
    XYZ n[3]; 
};

public:

/** \brief 计算出建立的三角形面的位置参数
 ** \param grid 小立方体的参数信息
 ** \param isolevel 给定的等值面
 ** \param triangles 建立的三角形的位置参数
 ** \return 在该小正方体内建立的三角形的个数
*/
int Polygonise(GRIDCELL grid,double isolevel,TRIANGLE *triangles){
   int i,ntriang;
   int cubeindex;
   XYZ vertlist[12];
   
   /*
    确定小立方体八个顶点的位置，是在点云的内部还是外部
   */
   cubeindex = 0;
   if (grid.val[0] < isolevel) cubeindex |= 1;
   if (grid.val[1] < isolevel) cubeindex |= 2;
   if (grid.val[2] < isolevel) cubeindex |= 4;
   if (grid.val[3] < isolevel) cubeindex |= 8;
   if (grid.val[4] < isolevel) cubeindex |= 16;
   if (grid.val[5] < isolevel) cubeindex |= 32;
   if (grid.val[6] < isolevel) cubeindex |= 64;
   if (grid.val[7] < isolevel) cubeindex |= 128;

   /* 判断小立方体是否完全在点云数据之内或是之外 */
   if (edgeTable[cubeindex] == 0)
      return(0);

   /* 找到三角面是从哪条边插入小立方体 */
   if (edgeTable[cubeindex] & 1)
      vertlist[0] =
         VertexInterp(isolevel,grid.p[0],grid.p[1],grid.val[0],grid.val[1]);
   if (edgeTable[cubeindex] & 2)
      vertlist[1] =
         VertexInterp(isolevel,grid.p[1],grid.p[2],grid.val[1],grid.val[2]);
   if (edgeTable[cubeindex] & 4)
      vertlist[2] =
         VertexInterp(isolevel,grid.p[2],grid.p[3],grid.val[2],grid.val[3]);
   if (edgeTable[cubeindex] & 8)
      vertlist[3] =
         VertexInterp(isolevel,grid.p[3],grid.p[0],grid.val[3],grid.val[0]);
   if (edgeTable[cubeindex] & 16)
      vertlist[4] =
         VertexInterp(isolevel,grid.p[4],grid.p[5],grid.val[4],grid.val[5]);
   if (edgeTable[cubeindex] & 32)
      vertlist[5] =
         VertexInterp(isolevel,grid.p[5],grid.p[6],grid.val[5],grid.val[6]);
   if (edgeTable[cubeindex] & 64)
      vertlist[6] =
         VertexInterp(isolevel,grid.p[6],grid.p[7],grid.val[6],grid.val[7]);
   if (edgeTable[cubeindex] & 128)
      vertlist[7] =
         VertexInterp(isolevel,grid.p[7],grid.p[4],grid.val[7],grid.val[4]);
   if (edgeTable[cubeindex] & 256)
      vertlist[8] =
         VertexInterp(isolevel,grid.p[0],grid.p[4],grid.val[0],grid.val[4]);
   if (edgeTable[cubeindex] & 512)
      vertlist[9] =
         VertexInterp(isolevel,grid.p[1],grid.p[5],grid.val[1],grid.val[5]);
   if (edgeTable[cubeindex] & 1024)
      vertlist[10] =
         VertexInterp(isolevel,grid.p[2],grid.p[6],grid.val[2],grid.val[6]);
   if (edgeTable[cubeindex] & 2048)
      vertlist[11] =
         VertexInterp(isolevel,grid.p[3],grid.p[7],grid.val[3],grid.val[7]);

   /* 建立三角形面 */
   ntriang = 0;
   for (i=0;triTable[cubeindex][i]!=-1;i+=3) {
      triangles[ntriang].p[0] = vertlist[triTable[cubeindex][i  ]];
      triangles[ntriang].p[1] = vertlist[triTable[cubeindex][i+1]];
      triangles[ntriang].p[2] = vertlist[triTable[cubeindex][i+2]];
      ntriang++;
   }

   return(ntriang);
}

/** \brief 线性插值计算曲面与小正方体边的交点位置
 ** \param isolevel 给定的等值面
 ** \param p1 小正方体交线上的点1
 ** \param p2 小正方体交线上的点2
 ** \param valp1 小正方体交线上的点1对应的值
 ** \param valp2 小正方体交线上的点2对应的值
 ** \return 计算后得到的交点坐标
*/
XYZ VertexInterp(double isolevel,XYZ p1,XYZ p2,double valp1,double valp2){
   double mu;
   XYZ p;

   if (ABS(isolevel-valp1) < 0.00001)
      return(p1);
   if (ABS(isolevel-valp2) < 0.00001)
      return(p2);
   if (ABS(valp1-valp2) < 0.00001)
      return(p1);
   mu = (isolevel - valp1) / (valp2 - valp1);
   p.x = p1.x + mu * (p2.x - p1.x);
   p.y = p1.y + mu * (p2.y - p1.y);
   p.z = p1.z + mu * (p2.z - p1.z);

   return(p);
}


};


