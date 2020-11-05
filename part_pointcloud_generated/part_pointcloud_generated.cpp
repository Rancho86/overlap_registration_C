#include<conio.h>
#include<vector>
#include<iostream>
#include<string>
#include<ctime>
#include<algorithm>
#include<math.h>
#include<omp.h> //并行计算
// io读取的头文件
#include <pcl/PolygonMesh.h>
#include <pcl/io/io.h>
#include<pcl/io/obj_io.h>
#include<pcl/io/ply_io.h>
#include<pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>//loadPolygonFile所属头文件；
#include<pcl/console/parse.h>

//
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>	
#include<pcl/filters/extract_indices.h>
using namespace std;


void printUsage(const char* progName)
{
	cout << "先输入一个点云名字，再加两个点云的百分比，三个旋转角度" << endl;
	cout << "例子：" << progName << " source_point_cloud.pcd/obj/ply 0.6 0.7 10 0 15" << endl;
	cout << "测试版本" << endl;
}


//读取点云文件//
void read_pointcloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud,int argc, char **argv)
{
	clock_t read_pc_start = clock();
	string pc1name = argv[1];
	string pc1ext = pc1name.substr(pc1name.size() - 3);
	pcl::PolygonMesh mesh1;
	if (pc1ext.compare("ply") == 0)
		pcl::io::loadPLYFile(argv[1], *source_point_cloud);
	if (pc1ext.compare("obj") == 0)
	{
		cout << "开始读source_point_cloud" << endl;
		pcl::io::loadPolygonFile(argv[1], mesh1);
		cout << "读完source_point_cloud，转pcl" << endl;
		pcl::fromPCLPointCloud2(mesh1.cloud, *source_point_cloud);
		cout << "处理完source_point_cloud" << endl;
	}
	//pcl::io::loadOBJFile(argv[1], *source_point_cloud); //这种读入的方式特别慢
	if (pc1ext.compare("pcd") == 0)
		pcl::io::loadPCDFile(argv[1], *source_point_cloud);
	
	//去除nan点
	std::vector<int> indices_source_nan_indice;
	pcl::removeNaNFromPointCloud(*source_point_cloud, *source_point_cloud, indices_source_nan_indice);

	clock_t read_pc_end = clock();
	cout << "read pointcloud time: " << (double)(read_pc_end - read_pc_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	cout << "source_point_cloud size : " << source_point_cloud->points.size() << endl;
}

Eigen::Matrix3f revertmatrix(double x, double y, double z) {
	Eigen::Matrix3f Rx;
	Eigen::Matrix3f Ry;
	Eigen::Matrix3f Rz;
	Eigen::Matrix3f R;
	Rx << 1, 0, 0,
		0, cos(x*3.1415 / 180), -sin(x*3.1415 / 180),
		0, sin(x*3.1415 / 180), cos(x*3.1415 / 180);
	//cout << Rx << endl;
	Ry << cos(y*3.1415 / 180), 0, sin(y*3.1415 / 180),
		0, 1, 0,
		-sin(y*3.1415 / 180), 0, cos(y*3.1415 / 180);
	Rz << cos(z*3.1415 / 180), -sin(z*3.1415 / 180), 0,
		sin(z*3.1415 / 180), cos(z*3.1415 / 180), 0,
		0, 0, 1;
	R = Rz * Ry*Rx;
	return R;
}


bool cmp (vector<double>& a, vector<double>&b)
	{
		return a[2] < b[2];//表示按第3列从小到大进行排序
	}

void generate_part_pointcloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr origin_point_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud_overlap, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud_overlap, Eigen::Matrix4f &R_4,int argc, char **argv)
{
	int origin_point_cloud_count = origin_point_cloud->points.size();
	//——————————————————————————————————————————————————————————————————————————————————————————————————————————————
	//把点云按照一个扫描方向（y轴）排序
	vector<vector<double>> originalpoints;
	for (int i = 0; i < origin_point_cloud_count; i++) {
		vector<double> points;
		points.push_back(origin_point_cloud->points[i].x);
		points.push_back(origin_point_cloud->points[i].y);
		points.push_back(origin_point_cloud->points[i].z);
		originalpoints.push_back(points);
	}
	sort(originalpoints.begin(), originalpoints.end(), cmp);
	for (int i = 0; i < origin_point_cloud_count; i++) {
		origin_point_cloud->points[i].x = originalpoints[i][0];
		origin_point_cloud->points[i].y = originalpoints[i][1];
		origin_point_cloud->points[i].z = originalpoints[i][2];
	}

	//——————————————————————————————————————————————————————————————————————————————————————————————————————————————
	std::vector<int> source_indices;
	std::vector<int> target_indices;
	std::vector<int> overlap_indices;
	double source_percent = atof(argv[2]);
	int source_count = floor(origin_point_cloud_count*source_percent);
	double target_percent = atof(argv[3]);
	int target_count = floor(origin_point_cloud_count*target_percent);
	for (int i = 0; i < source_count; i++)
		source_indices.push_back(i);
	for (int j = origin_point_cloud_count-target_count; j < origin_point_cloud_count; j++)
		target_indices.push_back(j);
	for (int i = origin_point_cloud_count-target_count;i< source_count;i++)
		overlap_indices.push_back(i);
	boost::shared_ptr<std::vector<int>> source_index_ptr = boost::make_shared<std::vector<int>>(source_indices);
	boost::shared_ptr<std::vector<int>> target_index_ptr = boost::make_shared<std::vector<int>>(target_indices);
	boost::shared_ptr<std::vector<int>> overlap_index_ptr = boost::make_shared<std::vector<int>>(overlap_indices);
	//pcl::IndicesPtr  source_index_ptr(new std::vector<int>(source_indices));
	//pcl::IndicesPtr  target_index_ptr(new std::vector<int>(target_indices));
    //利用ExtractIndices根据索引进行source点云的提取
    pcl::ExtractIndices<pcl::PointXYZRGBA> source_extract;
	source_extract.setInputCloud(origin_point_cloud);
	source_extract.setIndices(source_index_ptr);
	source_extract.setNegative(false);//如果设为true,可以提取指定index之外的点云
	source_extract.filter(*source_cloud);
	//利用ExtractIndices根据索引进行target点云的提取
	pcl::ExtractIndices<pcl::PointXYZRGBA> target_extract;
	target_extract.setInputCloud(origin_point_cloud);
	target_extract.setIndices(target_index_ptr);
	target_extract.setNegative(false);//如果设为true,可以提取指定index之外的点云
	target_extract.filter(*target_cloud);
	//利用ExtractIndices根据索引进行source_overlap点云的提取
	pcl::ExtractIndices<pcl::PointXYZRGBA> source_overlap_extract;
	source_overlap_extract.setInputCloud(origin_point_cloud);
	source_overlap_extract.setIndices(overlap_index_ptr);
	source_overlap_extract.setNegative(false);//如果设为true,可以提取指定index之外的点云
	source_overlap_extract.filter(*source_point_cloud_overlap);
	//利用ExtractIndices根据索引进行target_overlap点云的提取
	pcl::ExtractIndices<pcl::PointXYZRGBA> target_overlap_extract;
	target_overlap_extract.setInputCloud(origin_point_cloud);
	target_overlap_extract.setIndices(overlap_index_ptr);
	target_overlap_extract.setNegative(false);//如果设为true,可以提取指定index之外的点云
	target_overlap_extract.filter(*target_point_cloud_overlap);
	//变换一下target点云
	double x_r = atof(argv[4]);
	double y_r = atof(argv[5]);
	double z_r = atof(argv[6]);
	Eigen::Matrix3f R_3 = revertmatrix(x_r, y_r, z_r);
	//旋转矩阵求旋转角
	double rotation = fabs(acos((R_3.trace() - 1) / 2)) * 180 / 3.14;//旋转角
	cout << "绕x轴旋转：" << x_r << endl;
	cout << "绕y轴旋转：" << y_r << endl;
	cout << "绕z轴旋转：" << z_r << endl;
	cout << "总旋转角度：" << rotation << endl;
	R_4.block<3, 3>(0, 0) = R_3;
	R_4(3, 0) = 0;
	R_4(3, 1) = 0;
	R_4(3, 2) = 0;
	R_4(3, 3) = 1;
	R_4(0, 3) = 0;
	R_4(1, 3) = 0;
	R_4(2, 3) = 0;
	pcl::transformPointCloud(*target_cloud, *target_cloud, R_4);
	pcl::transformPointCloud(*target_point_cloud_overlap, *target_point_cloud_overlap, R_4);
}


int
main(int argc, char** argv)
{

  //——————————————————————————————————————————————————————	
	if (argc < 6)
	{
		PCL_ERROR("输入变量数量不足！\n");
		printUsage(argv[0]);
		return (0);
	}
	double overlap = atof(argv[2]) + atof(argv[3])-1;
	if (overlap < 0)
	{
		PCL_ERROR("点云没有重叠区域！第二个参数和第三个参数和要大于1\n");
		printUsage(argv[0]);
		return (0);
	}
	cout << "重叠区域大小为" << overlap<< endl;
	//——————————————————————————————————————————————————————	
	//定义输入的两个点云
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr origin_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // 源点云
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // source点云
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // target点云
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud_overlap(new pcl::PointCloud<pcl::PointXYZRGBA>); // source_overlap点云
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud_overlap(new pcl::PointCloud<pcl::PointXYZRGBA>); // target_overlap点云
	//——————————————————————————————————————————————————————	
	//读取输入的源点云
	read_pointcloud(origin_point_cloud, argc, argv);
	cout << "获取完原始点云" << endl;
	//——————————————————————————————————————————————————————
	//按照角度旋转生成点云
	Eigen::Matrix4f R_4;
	generate_part_pointcloud(origin_point_cloud, source_point_cloud, target_point_cloud,source_point_cloud_overlap, target_point_cloud_overlap,R_4,argc, argv);
	cout << "生成完点云" << endl;
	//——————————————————————————————————————————————————————
	//显示原始点云
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("原点云"));
	int v1(0);
	int v2(0);
	//source点云视角
	viewer1->createViewPort(0.0, 0.0, 0.5, 1.0, v1);//(Xmin,Ymin,Xmax,Ymax)设置不同视角窗口坐标
	viewer1->setBackgroundColor(255, 255, 255, v1);//设置背景色为白色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> orign_point_cloud_color_handler(origin_point_cloud, 0, 0, 0);//黑色
	viewer1->addText("orign_point_cloud", 10, 10, 1.0, 0.0, 0.0, "v1 text", v1);
	viewer1->addPointCloud(origin_point_cloud, orign_point_cloud_color_handler, "origin_point_cloud", v1);
	//target点云视角
	viewer1->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer1->setBackgroundColor(255, 255, 255, v2);//设置背景色为白色
	viewer1->addText("source and target_point_cloud", 10, 10, 1.0, 0.0, 0.0, "v2 text", v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> source_pointcloud_color_handler(source_point_cloud, 255,0, 0);//红色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> target_pointcloud_color_handler(target_point_cloud, 0, 255, 0);//绿色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> overlap_pointcloud_color_handler(target_point_cloud, 255, 255, 0);//黄色
	viewer1->addPointCloud(source_point_cloud, source_pointcloud_color_handler,"source_point_cloud", v2);
	viewer1->addPointCloud(target_point_cloud, target_pointcloud_color_handler,"target_point_cloud", v2);
	viewer1->addPointCloud(source_point_cloud_overlap, overlap_pointcloud_color_handler, "source_point_cloud_overlap", v2);
	viewer1->addPointCloud(target_point_cloud_overlap, overlap_pointcloud_color_handler, "target_point_cloud_overlap", v2);
	//分开显示生成的点云
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("生成的点云"));
	//source点云视角
	viewer2->createViewPort(0.0, 0.0, 0.5, 1.0, v1);//(Xmin,Ymin,Xmax,Ymax)设置不同视角窗口坐标
	viewer2->setBackgroundColor(255, 255, 255, v1);//设置背景色为白色
	viewer2->addText("source_point_cloud", 10, 10, 1.0, 0.0, 0.0, "v1source text", v1);
	viewer2->addText("source_point_cloud_overlap", 10, 30, 1.0, 1.0, 0.0, "v1overlap text", v1);
	//target点云视角
	viewer2->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer2->setBackgroundColor(255, 255, 255, v2);//设置背景色为白色
	viewer2->addText("target_point_cloud", 10, 10, 0.0, 1.0, 0.0, "v2target text", v2);
	viewer2->addText("target_point_cloud_overlap", 10, 30, 1.0, 1.0, 0.0, "v2overlap text", v2);
	viewer2->addPointCloud(source_point_cloud, source_pointcloud_color_handler, "source_point_cloud", v1);
	viewer2->addPointCloud(target_point_cloud, target_pointcloud_color_handler, "target_point_cloud", v2);
	viewer2->addPointCloud(source_point_cloud_overlap, overlap_pointcloud_color_handler, "source_point_cloud_overlap", v1);
	viewer2->addPointCloud(target_point_cloud_overlap, overlap_pointcloud_color_handler, "target_point_cloud_overlap", v2);
	string origin_point_cloud_name(argv[1]);
	string source_point_cloud_rate(argv[2]);
	string target_point_cloud_rate(argv[3]);
	string x_rotation(argv[4]);
	string y_rotation(argv[5]);
	string z_rotation(argv[6]);
	string source_point_cloud_filepath= origin_point_cloud_name.substr(0,origin_point_cloud_name.size() - 4) +"_"+ source_point_cloud_rate+"_0"+ "_0"+ "_0";
	string target_point_cloud_filepath = origin_point_cloud_name.substr(0,origin_point_cloud_name.size() - 4) + "_" + target_point_cloud_rate+ "_"+ x_rotation + "_" + y_rotation + "_" + z_rotation ;
	string transform_matrix_filepath = origin_point_cloud_name.substr(0, origin_point_cloud_name.size() - 4) + "_" + source_point_cloud_rate + "_" + target_point_cloud_rate + "_" + x_rotation + "_" + y_rotation + "_" + z_rotation + ".txt";
	//pcl::io::savePCDFileASCII(source_point_cloud_filepath+".pcd", *source_point_cloud);
	//pcl::io::savePCDFileASCII(target_point_cloud_filepath + ".pcd", *target_point_cloud);
	pcl::io::savePLYFileASCII(source_point_cloud_filepath + ".ply", *source_point_cloud);
	pcl::io::savePLYFileASCII(target_point_cloud_filepath + ".ply", *target_point_cloud);

	std::ofstream fout;
	fout.open(transform_matrix_filepath, std::ios::app);//在文件末尾追加写入
	fout << R_4 << std::endl;//每次写完一个矩阵以后换行
	fout.close();
	while (!viewer1->wasStopped())
	{

		viewer1->spinOnce(100);   //
		viewer2->spinOnce(100);
	}

  //——————————————————————————————————————————————————————
	return (0);
}
