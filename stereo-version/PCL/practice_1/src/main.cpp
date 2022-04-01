#include"pcl_lib.h"

using namespace pcl;

int mian(){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    cloud.reset(new pcl::PointCloud<pcl::PointXYZ> ());
    
    return 0;
}
