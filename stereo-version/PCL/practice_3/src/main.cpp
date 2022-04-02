#include"pcl_lib.h"

int main(){

    pcl::PCDReader reader;                  //PCD文件读取类
    pcl::PCLPointCloud2 org;                //用于接受PCD点云数据的结构

    // reader.read("../../doc/practice_1.pcd",org);

    pcl::io::loadPCDFile("../../doc/practice_1.pcd",org);   //对reader类的一个封装

    for(auto &f : org.fields)               //显示PCD点云数据类型
        std::cout << f.name <<std::endl;



    pcl::PointCloud<pcl::PointXYZRGB> cloud;    //这次没有取指针

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);

    pcl::fromPCLPointCloud2<pcl::PointXYZRGB>(org, cloud);  //将读取的PCD文件转换成可被可视化的点云数据格式

    pcl::PCDWriter writer;                                              //将文件存储
    writer.writeASCII("../../doc/practice_3_ascii_writer.pcd", cloud);
    writer.writeBinary("../../doc/practice_3_binary_writer.pcd", cloud);
    writer.writeBinary("../../doc/practice_3_binary_compress_writer.pcd", cloud);

    pcl::io::savePCDFileASCII("../../doc/practice_3_ascii_io.pcd", cloud);

    viewer->addPointCloud(cloud.makeShared());

    while(!viewer->wasStopped())
        viewer->spinOnce(100);


    return 0;

}
