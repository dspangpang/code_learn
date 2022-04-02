#define PCL_NO_PRECOMPILE  //取消预编译

#include"pcl_lib.h"

struct EIGEN_ALIGN16 MyPoint        //自定义的点云数据模板
{
    PCL_ADD_POINT4D                 //添加X，Y，Z坐标和对齐填充位
    PCL_ADD_RGB
    int time_stamp;              //自定义的类型
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW //eigen库作者提供的ALIGNED_OPERATOR_NEW
};

POINT_CLOUD_REGISTER_POINT_STRUCT(MyPoint,             // 把自定义的点云结构体注册到PCL库之中
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (std::uint32_t, rgb, rgb)
                                    (int, time_stamp, time_stamp)
    );

int getTimeStamp(){

    static int time = 0;
    time ++ ;
    return time ;
}


int main(){

    pcl::PointCloud<MyPoint>::Ptr cloud;
    cloud.reset(new pcl::PointCloud<MyPoint>);
    cloud->width = 100;
    cloud->height = 100;
    cloud->is_dense = false;
    cloud->points.resize(100 * 100);

    for (size_t i = 0; i < 100; i++)
    {
        for (size_t j = 0; j < 100; j++)
        {
            cloud->points[i * 100 + j].x = rand() / (RAND_MAX + 1.0f);  //没有重载二位数组
            cloud->points[i * 100 + j].y = rand() / (RAND_MAX + 1.0f);
            cloud->points[i * 100 + j].z = rand() / (RAND_MAX + 1.0f);

            cloud->points[i * 100 + j].r = rand()%256;
            cloud->points[i * 100 + j].g = rand()%256;
            cloud->points[i * 100 + j].b = rand()%256;

            cloud->points[i * 100 + j].time_stamp = getTimeStamp();
        }
        
    }

    pcl::io::savePCDFile("practice_2.pcd",*cloud);
    

    // to show
#if 0
    pcl::visualization::CloudViewer viewer("viewer");   
    viewer.showCloud(cloud);                    //不是模板函数实现的，是通过重载实现的
    return 0; 
#else
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer); //在预编译的时候容易找不到自建模板，所以可以取消预编译
    viewer->addPointCloud<MyPoint>(cloud);

    //viewer->setPointCloudRenderingProperties可以设置点的属性参数
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);  //图像100毫秒循环一次
    }
    
#endif 
    return 0; 
}
