#pragma once

#include"pcl_lib.h"


struct EIGEN_ALIGN16 MyPoint        //自定义的点云数据模板
{
    PCL_ADD_POINT4D                 //添加X，Y，Z坐标和对齐填充位
    PCL_ADD_RGB
    int time_stamp;              //自定义的类型
    
    inline MyPoint(){

        data[0] = 0.0;
        data[1] = 0.0;
        data[2] = 0.0;
        data[3] = 0.0;

        rgba = 0;

        time_stamp = 0;
    }
    
    inline MyPoint(float _x, float _y, float _z, std::uint32_t _rgba, int _time_stamp){
        
        x = _x;
        y = _y;
        z = _z;

        rgba = _rgba;

        time_stamp = _time_stamp;
    }
    
    friend std::ostream &operator<< (std::ostream &os, MyPoint &p){

        os << "(" << p.x << "," << p.y << "," << p.z << "," << p.rgba << "," << p.time_stamp << ")";

        return os;

    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //eigen库作者提供的ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(MyPoint,             // 把自定义的点云结构体注册到PCL库之中
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (std::uint32_t, rgb, rgb)
                                    (int, time_stamp, time_stamp)
    );
