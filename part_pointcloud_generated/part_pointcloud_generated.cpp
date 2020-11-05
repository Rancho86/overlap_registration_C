#include<conio.h>
#include<vector>
#include<iostream>
#include<string>
#include<ctime>
#include<algorithm>
#include<math.h>
#include<omp.h> //���м���
// io��ȡ��ͷ�ļ�
#include <pcl/PolygonMesh.h>
#include <pcl/io/io.h>
#include<pcl/io/obj_io.h>
#include<pcl/io/ply_io.h>
#include<pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>//loadPolygonFile����ͷ�ļ���
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
	cout << "������һ���������֣��ټ��������Ƶİٷֱȣ�������ת�Ƕ�" << endl;
	cout << "���ӣ�" << progName << " source_point_cloud.pcd/obj/ply 0.6 0.7 10 0 15" << endl;
	cout << "���԰汾" << endl;
}


//��ȡ�����ļ�//
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
		cout << "��ʼ��source_point_cloud" << endl;
		pcl::io::loadPolygonFile(argv[1], mesh1);
		cout << "����source_point_cloud��תpcl" << endl;
		pcl::fromPCLPointCloud2(mesh1.cloud, *source_point_cloud);
		cout << "������source_point_cloud" << endl;
	}
	//pcl::io::loadOBJFile(argv[1], *source_point_cloud); //���ֶ���ķ�ʽ�ر���
	if (pc1ext.compare("pcd") == 0)
		pcl::io::loadPCDFile(argv[1], *source_point_cloud);
	
	//ȥ��nan��
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
		return a[2] < b[2];//��ʾ����3�д�С�����������
	}

void generate_part_pointcloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr origin_point_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_cloud, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud_overlap, pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud_overlap, Eigen::Matrix4f &R_4,int argc, char **argv)
{
	int origin_point_cloud_count = origin_point_cloud->points.size();
	//����������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
	//�ѵ��ư���һ��ɨ�跽��y�ᣩ����
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

	//����������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������������
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
    //����ExtractIndices������������source���Ƶ���ȡ
    pcl::ExtractIndices<pcl::PointXYZRGBA> source_extract;
	source_extract.setInputCloud(origin_point_cloud);
	source_extract.setIndices(source_index_ptr);
	source_extract.setNegative(false);//�����Ϊtrue,������ȡָ��index֮��ĵ���
	source_extract.filter(*source_cloud);
	//����ExtractIndices������������target���Ƶ���ȡ
	pcl::ExtractIndices<pcl::PointXYZRGBA> target_extract;
	target_extract.setInputCloud(origin_point_cloud);
	target_extract.setIndices(target_index_ptr);
	target_extract.setNegative(false);//�����Ϊtrue,������ȡָ��index֮��ĵ���
	target_extract.filter(*target_cloud);
	//����ExtractIndices������������source_overlap���Ƶ���ȡ
	pcl::ExtractIndices<pcl::PointXYZRGBA> source_overlap_extract;
	source_overlap_extract.setInputCloud(origin_point_cloud);
	source_overlap_extract.setIndices(overlap_index_ptr);
	source_overlap_extract.setNegative(false);//�����Ϊtrue,������ȡָ��index֮��ĵ���
	source_overlap_extract.filter(*source_point_cloud_overlap);
	//����ExtractIndices������������target_overlap���Ƶ���ȡ
	pcl::ExtractIndices<pcl::PointXYZRGBA> target_overlap_extract;
	target_overlap_extract.setInputCloud(origin_point_cloud);
	target_overlap_extract.setIndices(overlap_index_ptr);
	target_overlap_extract.setNegative(false);//�����Ϊtrue,������ȡָ��index֮��ĵ���
	target_overlap_extract.filter(*target_point_cloud_overlap);
	//�任һ��target����
	double x_r = atof(argv[4]);
	double y_r = atof(argv[5]);
	double z_r = atof(argv[6]);
	Eigen::Matrix3f R_3 = revertmatrix(x_r, y_r, z_r);
	//��ת��������ת��
	double rotation = fabs(acos((R_3.trace() - 1) / 2)) * 180 / 3.14;//��ת��
	cout << "��x����ת��" << x_r << endl;
	cout << "��y����ת��" << y_r << endl;
	cout << "��z����ת��" << z_r << endl;
	cout << "����ת�Ƕȣ�" << rotation << endl;
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

  //������������������������������������������������������������������������������������������������������������	
	if (argc < 6)
	{
		PCL_ERROR("��������������㣡\n");
		printUsage(argv[0]);
		return (0);
	}
	double overlap = atof(argv[2]) + atof(argv[3])-1;
	if (overlap < 0)
	{
		PCL_ERROR("����û���ص����򣡵ڶ��������͵�����������Ҫ����1\n");
		printUsage(argv[0]);
		return (0);
	}
	cout << "�ص������СΪ" << overlap<< endl;
	//������������������������������������������������������������������������������������������������������������	
	//�����������������
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr origin_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // Դ����
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // source����
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // target����
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud_overlap(new pcl::PointCloud<pcl::PointXYZRGBA>); // source_overlap����
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud_overlap(new pcl::PointCloud<pcl::PointXYZRGBA>); // target_overlap����
	//������������������������������������������������������������������������������������������������������������	
	//��ȡ�����Դ����
	read_pointcloud(origin_point_cloud, argc, argv);
	cout << "��ȡ��ԭʼ����" << endl;
	//������������������������������������������������������������������������������������������������������������
	//���սǶ���ת���ɵ���
	Eigen::Matrix4f R_4;
	generate_part_pointcloud(origin_point_cloud, source_point_cloud, target_point_cloud,source_point_cloud_overlap, target_point_cloud_overlap,R_4,argc, argv);
	cout << "���������" << endl;
	//������������������������������������������������������������������������������������������������������������
	//��ʾԭʼ����
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("ԭ����"));
	int v1(0);
	int v2(0);
	//source�����ӽ�
	viewer1->createViewPort(0.0, 0.0, 0.5, 1.0, v1);//(Xmin,Ymin,Xmax,Ymax)���ò�ͬ�ӽǴ�������
	viewer1->setBackgroundColor(255, 255, 255, v1);//���ñ���ɫΪ��ɫ
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> orign_point_cloud_color_handler(origin_point_cloud, 0, 0, 0);//��ɫ
	viewer1->addText("orign_point_cloud", 10, 10, 1.0, 0.0, 0.0, "v1 text", v1);
	viewer1->addPointCloud(origin_point_cloud, orign_point_cloud_color_handler, "origin_point_cloud", v1);
	//target�����ӽ�
	viewer1->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer1->setBackgroundColor(255, 255, 255, v2);//���ñ���ɫΪ��ɫ
	viewer1->addText("source and target_point_cloud", 10, 10, 1.0, 0.0, 0.0, "v2 text", v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> source_pointcloud_color_handler(source_point_cloud, 255,0, 0);//��ɫ
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> target_pointcloud_color_handler(target_point_cloud, 0, 255, 0);//��ɫ
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> overlap_pointcloud_color_handler(target_point_cloud, 255, 255, 0);//��ɫ
	viewer1->addPointCloud(source_point_cloud, source_pointcloud_color_handler,"source_point_cloud", v2);
	viewer1->addPointCloud(target_point_cloud, target_pointcloud_color_handler,"target_point_cloud", v2);
	viewer1->addPointCloud(source_point_cloud_overlap, overlap_pointcloud_color_handler, "source_point_cloud_overlap", v2);
	viewer1->addPointCloud(target_point_cloud_overlap, overlap_pointcloud_color_handler, "target_point_cloud_overlap", v2);
	//�ֿ���ʾ���ɵĵ���
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("���ɵĵ���"));
	//source�����ӽ�
	viewer2->createViewPort(0.0, 0.0, 0.5, 1.0, v1);//(Xmin,Ymin,Xmax,Ymax)���ò�ͬ�ӽǴ�������
	viewer2->setBackgroundColor(255, 255, 255, v1);//���ñ���ɫΪ��ɫ
	viewer2->addText("source_point_cloud", 10, 10, 1.0, 0.0, 0.0, "v1source text", v1);
	viewer2->addText("source_point_cloud_overlap", 10, 30, 1.0, 1.0, 0.0, "v1overlap text", v1);
	//target�����ӽ�
	viewer2->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	viewer2->setBackgroundColor(255, 255, 255, v2);//���ñ���ɫΪ��ɫ
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
	fout.open(transform_matrix_filepath, std::ios::app);//���ļ�ĩβ׷��д��
	fout << R_4 << std::endl;//ÿ��д��һ�������Ժ���
	fout.close();
	while (!viewer1->wasStopped())
	{

		viewer1->spinOnce(100);   //
		viewer2->spinOnce(100);
	}

  //������������������������������������������������������������������������������������������������������������
	return (0);
}
