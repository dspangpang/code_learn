#include"pcl_lib.h"

using namespace pcl;

int main(){

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud->width = 1000;
    cloud->height =1;
    cloud->is_dense = false;
    cloud->points.resize(1000);
       
    for (auto i = 0; i < 1000; i++)
    {
        //xyz
        cloud->points[i].data[0] = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].data[2] = 1024 * rand() / (RAND_MAX + 1.0f);

        //rgb

        cloud->points[i].r = 1024 * rand() / (256);
        cloud->points[i].g = 1024 * rand() / (256);
        cloud->points[i].b = 1024 * rand() / (256);

    }
    
    pcl::io::savePCDFile("practice_1.pcd",*cloud);
    pcl::visualization::CloudViewer viewer("practice_1 view");
    viewer.showCloud(cloud);
    
    while(!viewer.wasStopped()); //在窗口关闭时结束程序

    return 0;

    
}
