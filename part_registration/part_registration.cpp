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
//�������õ���ͷ�ļ�
#include <pcl/filters/random_sample.h> 
#include<pcl/filters/extract_indices.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/uniform_sampling.h>
// ransac��׼��Ҫ��ͷ�ļ�
#include<pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
//��������ͷ�ļ�
#include <pcl/features/normal_3d.h>
//#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
using namespace std;


void printUsage(const char* progName)
{
	cout << "�����������������֣��ټ�һ����������ֿ��������ټ�һ��С������overlap����ռ��" << endl;
	cout << "���ӣ�"<<progName<<" source_point_cloud.pcd/obj/ply target_point_cloud.pcd/obj/ply 8 0.4" << endl;
	cout << "���԰汾" << endl;
}

//��ȡ�����ļ�//
void read_pointcloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud, int argc, char **argv)
{
	clock_t read_pc_start = clock();
	string pc1name = argv[1];
	string pc2name = argv[2];
	string pc1ext = pc1name.substr(pc1name.size() - 3);
	string pc2ext = pc2name.substr(pc2name.size() - 3);
	pcl::PolygonMesh mesh1;
	pcl::PolygonMesh mesh2;
	if (pc1ext.compare("ply") == 0)
		pcl::io::loadPLYFile(argv[1], *source_point_cloud);
	if (pc1ext.compare("obj") == 0)
	{
		cout << "��ʼ��source_point_cloud" <<  endl;
		pcl::io::loadPolygonFile(argv[1], mesh1);
		cout << "����source_point_cloud��תpcl" << endl;
		pcl::fromPCLPointCloud2(mesh1.cloud, *source_point_cloud);
		cout << "������source_point_cloud" << endl;
	}
		//pcl::io::loadOBJFile(argv[1], *source_point_cloud); //���ֶ���ķ�ʽ�ر���
	if (pc1ext.compare("pcd") == 0)
		pcl::io::loadPCDFile(argv[1], *source_point_cloud);
	if (pc2ext.compare("ply") == 0)
		pcl::io::loadPLYFile(argv[2], *target_point_cloud);
	if (pc2ext.compare("obj") == 0)		
	{
		cout << "��ʼ��target_point_cloud" << endl;
		pcl::io::loadPolygonFile(argv[2], mesh2);
		cout << "����target_point_cloud��תpcl" << endl;
		pcl::fromPCLPointCloud2(mesh2.cloud, *target_point_cloud);
		cout << "������target_point_cloud" << endl;
	}
		//pcl::io::loadOBJFile(argv[2], *target_point_cloud); //���ֶ���ķ�ʽ�ر���
	if (pc2ext.compare("pcd") == 0)
		pcl::io::loadPCDFile(argv[2], *target_point_cloud);
	
	//ȥ��nan��
	std::vector<int> indices_source_nan_indice;
	std::vector<int> indices_target_nan_indice;
	pcl::removeNaNFromPointCloud(*source_point_cloud, *source_point_cloud, indices_source_nan_indice);
	pcl::removeNaNFromPointCloud(*target_point_cloud, *target_point_cloud, indices_target_nan_indice);

	clock_t read_pc_end = clock();
	cout << "read pointcloud time: " << (double)(read_pc_end - read_pc_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	cout << "source_point_cloud size : " << source_point_cloud->points.size() << endl;
	cout << "target_point_cloud size : " << target_point_cloud->points.size() << endl;
}

void compute_pointcloud_density(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr point_cloud,double &delta) {
	int aa = point_cloud->points.size();
	double qxmax = point_cloud->points[0].x;
	double qxmin = point_cloud->points[0].x;
	double qymax = point_cloud->points[0].y;
	double qymin = point_cloud->points[0].y;
	double qzmax = point_cloud->points[0].z;
	double qzmin = point_cloud->points[0].z;
	for (int i = 0; i < aa - 1; i++)
	{
		double qx = point_cloud->points[i].x;
		qxmax = max(qx, qxmax);
		qxmin = min(qx, qxmin);
		double qy = point_cloud->points[i].y;
		qymax = max(qy, qymax);
		qymin = min(qy, qymin);
		double qz = point_cloud->points[i].z;
		qzmax = max(qz, qzmax);
		qzmin = min(qz, qzmin);

	}
	double pointnumber = pow(aa, 1.0 / 3);
	delta = max(max((qxmax - qxmin) / pointnumber, (qymax - qymin) / pointnumber), (qzmax - qzmin) / pointnumber);
}

//������
void DownSample(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	pcl::RandomSample<pcl::PointXYZRGBA> sor;
	sor.setInputCloud(cloud);
	sor.setSample(10000);
	sor.filter(*cloud);
}
//FPS
void ComputeFPS(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, const int nums, vector<int> &SampleIndex)
{
	// �����������û�����
	int sample_nums = nums;
	// ��¼���ָ�����е�����
	int cloud_points_size = cloud->points.size();
	// ��¼ʣ����Ƶ�����
	vector<int> rest_cloud_pointsIndex;
	// ��¼���ָ������ÿ���㵽�ָ���Ƽ��ϵľ���
	vector<double> rest2sample_dist(cloud_points_size);
	// ��ʼ��
	for (int i = 0; i < cloud_points_size; i++)
		rest_cloud_pointsIndex.push_back(i);
	// ����
	//cout << "rest_source_pointsIndex��������" << rest_cloud_pointsIndex.size() << endl;

	double max_dist = -999.0;         // ��¼�����룬ѡȡ�ڶ�����ʱʹ��
	double max_dist_;                 // ��¼�����룬ѡȡ�ڶ���֮��ĵ�ʱʹ��
	int farthest_point = -1;          // ��¼��Զ��
	double length_dist=DBL_MAX;               // �����֮��ľ��� 
	int sample_inteIndex = -1;           // ������������Էֶ�λ��
	int sample_inteIndex_value = -1;     // �������������λ�ö�Ӧ��ʵ���Ƶ�����
	int inteIndex_value = -1;         // ʣ�����������Ӧ��ʵ���Ƶ�����
	srand(time(NULL));                // ��ϵͳʱ���ʼ���������

	// Step1: ���ѡ��һ����
	clock_t  sample_start, sample_stop;
	sample_start = clock();
	// �������ѡ����ָ���Ƶ�һ��(�������������)
	int first_select = rand() % (rest_cloud_pointsIndex.size() + 1);
	int first_select_value = rest_cloud_pointsIndex[first_select];
	// ����һ�������������ָ����
	SampleIndex.push_back(rest_cloud_pointsIndex[first_select]);
	// ����
	//cout << SampleIndex.size() << endl;
	// ���ָ������ɾ���õ�
	vector<int>::iterator iter = rest_cloud_pointsIndex.begin() + first_select;
	rest_cloud_pointsIndex.erase(iter);
	// ����
	//cout << rest_cloud_pointsIndex.size() << endl;

	// Step2: ѡ��ڶ�����
	// ����ʣ������һ����ľ���
	for (size_t j = 0; j < rest_cloud_pointsIndex.size(); j++)
	{
		inteIndex_value = rest_cloud_pointsIndex[j];
		// �������
		length_dist = pow(abs(cloud->points[inteIndex_value].x - cloud->points[first_select_value].x), 2) +
			pow(abs(cloud->points[inteIndex_value].y - cloud->points[first_select_value].y), 2) +
			pow(abs(cloud->points[inteIndex_value].z - cloud->points[first_select_value].z), 2);

		rest2sample_dist[j] = length_dist;              // ��������������ÿ���㵽�������Ƶ���С���뱣��
		if (length_dist > max_dist)
		{
			max_dist = length_dist;
			farthest_point = j;
		}
	}
	// ���ڶ�����Զ��������������ָ����
	SampleIndex.push_back(rest_cloud_pointsIndex[farthest_point]);
	// ���õ��ʣ�������ɾ��
	iter = rest_cloud_pointsIndex.begin() + farthest_point;
	rest_cloud_pointsIndex.erase(iter);

	// ����
	//cout << "ѡȡ�ڶ�����ɹ�" << endl;
	//cout << SampleIndex.size() << endl;
	//cout << rest_cloud_pointsIndex.size() << endl;

	// Step3: ѡ�������ĵ㣬     �ȡ�min��,��max��
	while (SampleIndex.size() < sample_nums)
	{
		max_dist_ = -99999.0;                // ÿѡһ���㣬���丳ֵ��С���Ա����ʹ��

		// ����ʣ����� 
		for (int j = 0; j < rest_cloud_pointsIndex.size(); j++)
		{
			length_dist = DBL_MAX;
			inteIndex_value = rest_cloud_pointsIndex[j];
			// ����ʣ���������ѡȡ��֮��ľ��룬����֮ǰѡȡ�ĵ�֮�����Ƚ�ȡ��Сֵ�����¾������
			// �������

			for (int i = 0; i < SampleIndex.size(); i++) {
				sample_inteIndex_value = SampleIndex[i];
				double length_dist_a = pow(abs(cloud->points[inteIndex_value].x - cloud->points[sample_inteIndex_value].x), 2) +
					pow(abs(cloud->points[inteIndex_value].y - cloud->points[sample_inteIndex_value].y), 2) +
					pow(abs(cloud->points[inteIndex_value].z - cloud->points[sample_inteIndex_value].z), 2);
				length_dist = min(length_dist, length_dist_a);
			}

			// �ȽϾ��벢���¾���
				rest2sample_dist[j] = length_dist;


			// ��ȡ�����
				if (rest2sample_dist[j] > max_dist_)
				{
					farthest_point = j;
					max_dist_ = rest2sample_dist[j];
				}
		}
		// ��ѡȡ����Զ�㱣�����ָ������
		SampleIndex.push_back(rest_cloud_pointsIndex[farthest_point]);
		// ���õ��ʣ�������ɾ��
		iter = rest_cloud_pointsIndex.begin() + farthest_point;
		rest_cloud_pointsIndex.erase(iter);

		// ����
		//cout << "                 ѡȡһ����Զ��ɹ���" << endl;
	}
	sample_stop = clock();

	// ����
	cout << "������ϣ�" << endl;
	cout << "����������" << SampleIndex.size() << endl;
	cout << "��Զ�����(FPS)��ʱ��" << (double)(sample_stop - sample_start) / (double)CLOCKS_PER_SEC << "��" << endl;
}

// ---------------------------------------------------------------------------
// -----------------------------�ָ���Ƴ��� ---------------------------------
// ---------ʹ��FPS�����ĵ�Ϊ���ĵ㣬����KD�������и���� --------------------
// -------------ÿ�ηָ�һ������ --------------------------------------------
// ---------------------------------------------------------------------------
void SegmentCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA> &newcloud, vector<int> &pointIdxNKNSearch, const int index, const int seg_nums, const int K)
{
	clock_t  seg_start, seg_stop;
	seg_start = clock();
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZRGBA searchPoint = cloud->points[index];

	//int K = (int)((int)(cloud->points.size()*0.4) / seg_nums);

	//int K = 2000;  //���KӦ�ø��ݵ����ܶȾ�������������������ͬ����

	// ����ʹ��Kֵ��ѯ���ڵ㣬������
	//vector<int> pointIdxNKNSearch(K);
	vector<float> pointNKNSquaredDistance(K);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud < pcl::PointXYZRGBA>);
	pcl::ExtractIndices<pcl::PointXYZRGBA> extract;

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		boost::shared_ptr<vector<int>> pointIdxRadiusSearch_ptr = boost::make_shared<vector<int>>(pointIdxNKNSearch);
		// ����������ȡ����
		extract.setInputCloud(cloud);
		extract.setIndices(pointIdxRadiusSearch_ptr);
		extract.setNegative(false);                    //�����Ϊtrue, ������ȡָ��index֮��ĵ���
		extract.filter(*cloud_ptr);
		newcloud = *cloud_ptr;
	}
	seg_stop = clock();
	//cout << " �ָ���ƿ��е��������" << pointIdxNKNSearch.size() << endl;
	//cout << "�и�һƬ������ʱ��" << seg_stop - seg_start << "��" << endl;
}

// ---------------------------------------------------------------------------------------
// ------------------------------- ransac��׼ --------------------------------------------
// ���룺��Ƭ����׼����
// ������任����
void Ransac_registration(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud,
	Eigen::Matrix4f &transformation, double &score)
{	
	clock_t  ransac_start, ransac_end;
	clock_t  density_start, density_end;
	clock_t  downsample_start, downsample_end;
	clock_t  normal_start, normal_end;
	clock_t  fpfh_start, fpfh_end;
	clock_t  sac_start, sac_end;

	ransac_start = clock();
	//�����ܶ�
	density_start = clock();
	double sourcepoint_leafsize;
	compute_pointcloud_density(source_point_cloud, sourcepoint_leafsize);
	double targetpoint_leafsize;
	compute_pointcloud_density(target_point_cloud, targetpoint_leafsize);
	//cout << "sourcepoint_leafsize:" << sourcepoint_leafsize << endl;
	//cout << "targetpoint_leafsize:" << targetpoint_leafsize << endl;
	density_end = clock();
	//cout << "density��ʱ:" << (double)(density_end - density_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	//������		
	downsample_start = clock();
	pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_grid;
	voxel_grid.setLeafSize(sourcepoint_leafsize * 2, sourcepoint_leafsize * 2, sourcepoint_leafsize * 2);
	//voxel_grid.setLeafSize(0.01, 0.01, 0.01);
	voxel_grid.setInputCloud(source_point_cloud);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr SourceCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	voxel_grid.filter(*SourceCloud);

	pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_grid_2;
	voxel_grid_2.setLeafSize(targetpoint_leafsize*2, targetpoint_leafsize * 2, targetpoint_leafsize * 2);
	//voxel_grid_2.setLeafSize(0.01, 0.01,0.01);
	voxel_grid_2.setInputCloud(target_point_cloud);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr TargetCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	voxel_grid_2.filter(*TargetCloud);
	downsample_end = clock();
	//cout << "��������ʱ:" << (double)(downsample_end - downsample_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	
	//������淨��
	normal_start = clock();
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne_src;
	ne_src.setInputCloud(SourceCloud);
	pcl::search::KdTree< pcl::PointXYZRGBA>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZRGBA>());
	ne_src.setSearchMethod(tree_src);
	pcl::PointCloud<pcl::Normal>::Ptr SourceCloud_normals(new pcl::PointCloud< pcl::Normal>);
	ne_src.setRadiusSearch(sourcepoint_leafsize*6);
	//ne_src.setRadiusSearch(0.03);
	ne_src.compute(*SourceCloud_normals);

	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne_tgt;
	ne_tgt.setInputCloud(TargetCloud);
	pcl::search::KdTree< pcl::PointXYZRGBA>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZRGBA>());
	ne_tgt.setSearchMethod(tree_tgt);
	pcl::PointCloud<pcl::Normal>::Ptr TargetCloud_normals(new pcl::PointCloud< pcl::Normal>);
	//ne_tgt.setKSearch(20);
	ne_tgt.setRadiusSearch(targetpoint_leafsize*6);
	//ne_tgt.setRadiusSearch(0.03);
	ne_tgt.compute(*TargetCloud_normals);
	normal_end = clock();
	//cout << "normal��ʱ:" << (double)(normal_end - normal_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	
	//����FPFH
	fpfh_start = clock();
	pcl::FPFHEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh_src;
	fpfh_src.setInputCloud(SourceCloud);
	fpfh_src.setInputNormals(SourceCloud_normals);
	pcl::search::KdTree< pcl::PointXYZRGBA>::Ptr tree_src_fpfh(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	fpfh_src.setSearchMethod(tree_src_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_src.setRadiusSearch(targetpoint_leafsize * 10);
	//fpfh_src.setRadiusSearch(0.05);
	fpfh_src.compute(*fpfhs_src);
	//std:://cout << "compute *SourceCloud fpfh" << endl;

	pcl::FPFHEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh_tgt;
	fpfh_tgt.setInputCloud(TargetCloud);
	fpfh_tgt.setInputNormals(TargetCloud_normals);
	pcl::search::KdTree< pcl::PointXYZRGBA>::Ptr tree_tgt_fpfh(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_tgt.setRadiusSearch(targetpoint_leafsize * 10);
	//fpfh_tgt.setRadiusSearch(0.05);
	fpfh_tgt.compute(*fpfhs_tgt);
	//std:://cout << "compute *TargetCloud fpfh" << endl;
	fpfh_end = clock();
	//cout << "fpfh��ʱ:" << (double)(fpfh_end - fpfh_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	
	//SAC��׼
	sac_start = clock();
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::FPFHSignature33> scia;
	scia.setInputSource(SourceCloud);
	scia.setInputTarget(TargetCloud);
	scia.setSourceFeatures(fpfhs_src);
	scia.setTargetFeatures(fpfhs_tgt);
	//scia.setNumberOfSamples(20);
	//scia.setRANSACIterations(30);
	//scia.setRANSACOutlierRejectionThreshold(targetpoint_leafsize * 10);
	//scia.setCorrespondenceRandomness(20);
	//scia.setMinSampleDistance(1);
	//scia.setNumberOfSamples(2);
	//scia.setCorrespondenceRandomness(20);
	//PointCloud::Ptr sac_result(new PointCloud);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
	scia.align(*sac_result);
	//std:://cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	transformation = scia.getFinalTransformation();
	//std:://cout << transformation << endl;
	sac_end = clock();
	//cout << "sac��ʱ:" << (double)(sac_end - sac_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	ransac_end = clock();
	//cout << "ransac��ʱ:" << (double)(ransac_end - ransac_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	////cout << "ʹ��ransac�㷨��׼��ʱ��" << (ransac_stop - ransac_start) << "��" << endl;
	if (scia.hasConverged())
		score = scia.getFitnessScore();

}
void method_icp(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr OutCloud, Eigen::Matrix4f &registration_matrix)
{
	if (source_point_cloud->points.size() > 100000)
		DownSample(source_point_cloud);
	if (target_point_cloud->points.size() > 100000)
		DownSample(target_point_cloud);
	pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
	icp.setInputSource(source_point_cloud);
	icp.setInputTarget(target_point_cloud);
	pcl::PointCloud<pcl::PointXYZRGBA> Final;
	icp.align(Final);
	cout << "has converged:" << icp.hasConverged() << " score: " <<
		icp.getFitnessScore() << endl;
	cout << icp.getFinalTransformation() << endl;
	pcl::transformPointCloud(*source_point_cloud, *OutCloud, icp.getFinalTransformation());
	registration_matrix = icp.getFinalTransformation();
}

void method_sac_icp(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud,  Eigen::Matrix4f &registration_matrix)
{
	double sourcepoint_leafsize;
	compute_pointcloud_density(source_point_cloud, sourcepoint_leafsize);
	double targetpoint_leafsize;
	compute_pointcloud_density(target_point_cloud, targetpoint_leafsize);
	vector<int> indices_src; //����ȥ���ĵ������
	pcl::removeNaNFromPointCloud(*source_point_cloud, *source_point_cloud, indices_src);
	//cout << "remove *source_point_cloud nan" << endl;
	//�²����˲�
	pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_grid;
	voxel_grid.setLeafSize(sourcepoint_leafsize*2, sourcepoint_leafsize * 2, sourcepoint_leafsize * 2);
	voxel_grid.setInputCloud(source_point_cloud);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZRGBA>);
	voxel_grid.filter(*cloud_src);
	//cout << "down size *source_point_cloud from " << source_point_cloud->size() << "to" << cloud_src->size() << endl;
	//������淨��
	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne_src;
	ne_src.setInputCloud(cloud_src);
	pcl::search::KdTree< pcl::PointXYZRGBA>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZRGBA>());
	ne_src.setSearchMethod(tree_src);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
	ne_src.setRadiusSearch(sourcepoint_leafsize * 6);
	ne_src.compute(*cloud_src_normals);

	vector<int> indices_tgt;
	pcl::removeNaNFromPointCloud(*target_point_cloud, *target_point_cloud, indices_tgt);
	//cout << "remove *target_point_cloud nan" << endl;

	pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_grid_2;
	voxel_grid_2.setLeafSize(targetpoint_leafsize * 2, targetpoint_leafsize * 2, targetpoint_leafsize * 2);
	voxel_grid_2.setInputCloud(target_point_cloud);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_tgt(new pcl::PointCloud<pcl::PointXYZRGBA>);
	voxel_grid_2.filter(*cloud_tgt);
	//cout << "down size *target_point_cloud.pcd from " << target_point_cloud->size() << "to" << cloud_tgt->size() << endl;

	pcl::NormalEstimation<pcl::PointXYZRGBA, pcl::Normal> ne_tgt;
	ne_tgt.setInputCloud(cloud_tgt);
	pcl::search::KdTree< pcl::PointXYZRGBA>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZRGBA>());
	ne_tgt.setSearchMethod(tree_tgt);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
	//ne_tgt.setKSearch(20);
	ne_tgt.setRadiusSearch(targetpoint_leafsize * 6);
	ne_tgt.compute(*cloud_tgt_normals);

	//����FPFH
	pcl::FPFHEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh_src;
	fpfh_src.setInputCloud(cloud_src);
	fpfh_src.setInputNormals(cloud_src_normals);
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_src_fpfh(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	fpfh_src.setSearchMethod(tree_src_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_src.setRadiusSearch(sourcepoint_leafsize * 10);
	fpfh_src.compute(*fpfhs_src);
	//cout << "compute *cloud_src fpfh" << endl;

	pcl::FPFHEstimationOMP<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh_tgt;
	fpfh_tgt.setInputCloud(cloud_tgt);
	fpfh_tgt.setInputNormals(cloud_tgt_normals);
	pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_tgt_fpfh(new pcl::search::KdTree<pcl::PointXYZRGBA>);
	fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
	fpfh_tgt.setRadiusSearch(targetpoint_leafsize * 10);
	fpfh_tgt.compute(*fpfhs_tgt);
	cout << "compute *cloud_tgt fpfh" << endl;

	//SAC��׼
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::FPFHSignature33> scia;
	scia.setInputSource(cloud_src);
	scia.setInputTarget(cloud_tgt);
	scia.setSourceFeatures(fpfhs_src);
	scia.setTargetFeatures(fpfhs_tgt);
	//scia.setNumberOfSamples(20);
	//scia.setRANSACIterations(30);
	//scia.setRANSACOutlierRejectionThreshold(targetpoint_leafsize * 10);
	//scia.setCorrespondenceRandomness(20);
	//scia.setCorrespondenceRandomness(10);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
	scia.align(*sac_result);
	cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	Eigen::Matrix4f sac_trans;
	sac_trans = scia.getFinalTransformation();
	cout << sac_trans << endl;
	clock_t sac_time = clock();

	//icp��׼
	pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
	icp.setMaxCorrespondenceDistance(0.04);
	// ����������
	icp.setMaximumIterations(50);
	// ���α仯����֮��Ĳ�ֵ
	icp.setTransformationEpsilon(1e-10);
	// �������
	icp.setEuclideanFitnessEpsilon(0.2);
	//icp.setInputSource(sac_result);
	icp.setInputSource(cloud_src);
	icp.setInputTarget(cloud_tgt);
	pcl::PointCloud<pcl::PointXYZRGBA> Final;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
	//icp.align(Final);
	icp.align(*icp_result,sac_trans);//��ǰ��һ������׼����
	cout << "icp has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << endl;
	cout << icp.getFinalTransformation() << endl;
	Eigen::Matrix4f icp_trans= icp.getFinalTransformation();
	//registration_matrix =  icp_trans* sac_trans; //pcl::transform�Ǿ����ҳ˵��ƣ������ȱ任�ľ������Ҳ�
	registration_matrix = icp_trans;
	//if (icp.hasConverged())
		//score = icp.getFitnessScore();
}



// ---------------------------------------------------------------------------------------
// --------------------------------- ������Ƭ����֮���RMSֵ -----------------------------
// ���룺 ��Ƭ����   ()
// ���أ� ��Ƭ���Ƶ�RMSֵ(double)
void ComputeRMS(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr first_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr second_cloud, double &rms)
{
	clock_t  rms_start, rms_stop;
	rms_start = clock();

	double sums = 0;
	int count = 0;
	// ����Ƭ�����еĵ�������һ�£�ֻ����ͬ�������ĵ�֮���RMSֵ
	for (; (count < first_cloud->points.size()) && (count < second_cloud->points.size()); count++)
	{
		sums += pow((first_cloud->points[count].x - second_cloud->points[count].x), 2) +
			pow((first_cloud->points[count].y - second_cloud->points[count].y), 2) +
			pow((first_cloud->points[count].z - second_cloud->points[count].z), 2);
	}

	rms_stop = clock();
	//cout << "����RMSֵ��ʱ��" << (rms_stop - rms_start) << "��" << endl;

	rms = sqrt(sums / count);
}

// -----------------------------------------------------------------------------------------------
// ----��ѡȡrmsֵ��С��10����Ӧ��TR����Ȼ�����ÿ����ʣ��Ĳ�ֵ��
// ----ѡȡ��ֵ��С��TR��������Ϊ������Ƭ���Ƶ�TR���������׼ ----------------------------------
// ----------------------����汾��������ֵ����������С ------------------------------------------
// ���룺rmsֵ��������TR���������
// �������������TR��������ֵ
int ComputerRelativeTR_Threshold(const vector<double> rms_vector, const vector<Eigen::Matrix4f> TR_vector, double pointdis,vector<vector<int>>regionpair)
{
	vector<double> rms_copy;
	// ��ʼ��
	for (int i = 0; i < rms_vector.size(); i++)
		rms_copy.push_back(rms_vector[i]);

	// ����rms��Ӧ������������
	vector<int> rms_copy_index;
	// ��ʼ��
	for (int j = 0; j < rms_vector.size(); j++)
		rms_copy_index.push_back(j);

	vector<int> rms_min_index;              // ��¼�õ���ǰ10����Сrmsֵ����
	Eigen::Matrix4f TR_Diff;                // ��¼TR��ֵ
	Eigen::Matrix3f TR_Diff_3;
	int candidate_num = min(int(ceil(rms_vector.size()*0.5)), 10);
	vector<vector<int>> similartPair(candidate_num);
	vector<double> count_vector;                 // ��¼С����ֵ������������

	// ����ǰrms_vector.size()*0.5����Сrmsֵ����
	for (int i = 0; i < candidate_num; i++)        // ����10��
	{
		double rms_Min = DBL_MAX;          // Ϊ����rms��Сֵ������һ����ʼrms_Mim��׼ֵ
		int Min_Index = 0;
		for (int j = 0; j < rms_copy.size(); j++)      // ������������
		{
			if (rms_copy[j] < rms_Min)
			{
				rms_Min = rms_copy[j];
				Min_Index = j;
			}
		}
		//����rms��Сֵ������
		rms_min_index.push_back(rms_copy_index[Min_Index]);

		// ��Ŀǰ����Сֵɾ��
		vector<double>::iterator iter1 = rms_copy.begin() + Min_Index;
		rms_copy.erase(iter1);
		// ��Ŀǰ����Сֵ��Ӧ������ɾ��
		vector<int>::iterator iter2 = rms_copy_index.begin() + Min_Index;
		rms_copy_index.erase(iter2);

		// ����
		//cout << rms_copy.size() << endl;
	}

	// �����¼��ת�����ֵ�ı���
	double rotation_dif = 0.0;
	// �����¼ƽ�ƾ���ľ�ֵ�ı���
	double translation_dif = 0.0;

	// ������ת������ֵ
	double rotation_threshold = 15;//С30��jiu
	cout << "rotation_threshold:" << rotation_threshold << endl;
	// ������תƽ�ƾ�����ֵ
	double translation_threshold = pointdis * 20;
	cout << "translation_threshold:" << translation_threshold << endl;
	// ����ÿ��TR��ʣ��TR֮��Ĳ�ֵ
	for (int m = 0; m < candidate_num; m++)
	{
		//�����¼С����ֵ�ĸ����ı���
		int count = 0;

		// ����
		//cout << "��" << m << "��TR����"<< TR_vector[rms_min_index[m]] << endl;
		for (int n = 0; n < candidate_num; n++)
		{
			if (n != m)
			{
				//��������׼����֮�����ת����
				TR_Diff = TR_vector[rms_min_index[m]].inverse() * (TR_vector[rms_min_index[n]]);//������Ϊԭ������ת�õ� ����ת�÷�����ǰ��
				//cout << "��" << n << "��TR_Diff" << TR_Diff << endl;
				Eigen::Matrix3f TR_Diff_R;
				TR_Diff_R = TR_Diff.block<3, 3>(0, 0);
				//cout << "��" << n << "��TR_Diff_R" << TR_Diff_R << endl;
				//��ת��������ת��
				rotation_dif=fabs(acos((TR_Diff_R.trace() - 1) / 2)) * 180 / 3.14;
				// ����
				cout << "�͵�" << n << "��RT�������ת�ǶȲ�ֵ��" << rotation_dif << endl;

				// ���ֵTR�����ƽ�ƾ���ľ�ֵ
				translation_dif= (fabs(TR_Diff(0, 3)) + fabs(TR_Diff(1, 3)) + fabs(TR_Diff(2, 3))) / 3;

				// ����
				cout << "�͵�" << n << "��RT�����ƽ�Ʋ�ֵ��" << translation_dif << endl;

			

				// ��������ֵ���Ӧ��ֵ���бȽ�
				if ((rotation_dif < rotation_threshold) && (translation_dif < translation_threshold))
				{
						count += 1;
						similartPair[m].push_back(n);
				}
					
			}
		}
		// ����
		// ��ʾ��rmsֵ
	    cout << "�����"<< regionpair[rms_min_index[m]][0]<<"��"<< regionpair[rms_min_index[m]][1] <<"��RMSֵΪ" << rms_vector[rms_min_index[m]] << endl;
		cout << "�����������" << count << endl;
		count_vector.push_back(count);
	}

	// ��С����ֵ��������Ӧ��rms_min_index������
	int count_max = -1;             // ����һ����׼
	int count_max_index = 0;
	for (int k = 0; k < candidate_num; k++)
	{
		if (count_vector[k] > count_max)
		{
			count_max = count_vector[k];
			count_max_index = k;
		}
	}
	// ����
	cout << "���������" << count_vector[count_max_index] << endl;
	//similartPair[count_max_index];
	//cout << "��Ӧ��������" << count_max_index << endl;
	if (count_vector[count_max_index] > 1)
		return rms_min_index[count_max_index];
	else
		return rms_min_index[0];
}

int
 main (int argc, char** argv)
{
//������������������������������������������������������������������������������������������������������������	
	//�ֶ����ò�����
	double overlap = 0.4;     //�ص������С
	if(argc>=5)
		overlap = atof(argv[4]);
	int seg_nums = floor(1/overlap*5);         // �и���Ƶ�������Ĭ����11
	cout << "�����ص���������Ƽ�ʹ�÷ֿ�����Ϊ" << seg_nums << endl;
	if (argc >= 4)
		seg_nums = max(seg_nums,atoi(argv[3]));
	cout << "����ʹ�÷ֿ�����Ϊ" << seg_nums << endl;
	double K_factor = overlap*seg_nums;     //�ָ��ʱ����Ҫ���õ�Ӱ�����ӣ�����Kd���Ĵ�С
	int sac_times = 500;        //SAC����������
	double thresh = 0.001;      //SACɸѡ��ľ�����ֵ
	//int threads_num = min(omp_get_max_threads(), seg_nums);
	//cout << "ʹ���߳�����" << threads_num << endl;
	//omp_set_num_threads(threads_num);
	//omp_set_num_threads(omp_get_max_threads());
  //������������������������������������������������������������������������������������������������������������	
	if (argc < 3)
	{
		PCL_ERROR("��������������㣡\n");
		printUsage(argv[0]);
		return (0);
	}
  //������������������������������������������������������������������������������������������������������������	
  //�����������������
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // Դ����
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // Ŀ�����
  //������������������������������������������������������������������������������������������������������������	
  //��ȡ�������������
  read_pointcloud(source_point_cloud, target_point_cloud, argc, argv);
  string pc1name = argv[1];
  string pc2name = argv[2];
  if (source_point_cloud->points.size() > 50000)
	  DownSample(source_point_cloud);
  if (target_point_cloud->points.size() > 50000)
	  DownSample(target_point_cloud);
  //��������ܶ�
  double source_point_cloud_density=0.0;
  double target_point_cloud_density=0.0;
  compute_pointcloud_density(source_point_cloud, source_point_cloud_density);
  compute_pointcloud_density(target_point_cloud, target_point_cloud_density);
  cout << "source_point_cloud�����ܶ�:"<< source_point_cloud_density << endl;
  cout << "target_point_cloud�����ܶ�:" << target_point_cloud_density << endl;
  //������������������������������������������������������������������������������������������������������������
  //��ʾԭʼ����
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("ԭ����"));
  int v1(0);
  int v2(0);
  //source�����ӽ�
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> source_cloud_color_handler(source_point_cloud, 0, 255, 0);//������
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> target_cloud_color_handler(target_point_cloud, 0, 255, 0);//������
  viewer1->createViewPort(0.0, 0.0, 0.5, 1.0, v1);//(Xmin,Ymin,Xmax,Ymax)���ò�ͬ�ӽǴ�������
  viewer1->setBackgroundColor(255, 255, 255, v1);//���ñ���ɫΪ��ɫ
  viewer1->addText("source_point_cloud_image", 10, 10,1.0,0.0,0.0, "v1 text", v1);
  viewer1->addPointCloud(source_point_cloud, source_cloud_color_handler,"source_point_cloud",v1);
  //target�����ӽ�
  viewer1->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer1->setBackgroundColor(255, 255, 255, v2);//���ñ���ɫΪ��ɫ
  viewer1->addText("target_point_cloud_image", 10, 10, 1.0, 0.0, 0.0, "v2 text", v2);
  viewer1->addPointCloud(target_point_cloud, target_cloud_color_handler, "target_point_cloud",v2);
  //viewer2->addCoordinateSystem(1.0);
 
//������������������������������������������������������������������������������������������������������������
  //���Ʒֿ����ʾ
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("source���Ʒֿ��ӽ�"));
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer3(new pcl::visualization::PCLVisualizer("target���Ʒֿ��ӽ�"));
  viewer2->setBackgroundColor(255, 255, 255);//���ñ���ɫΪ��ɫ
  viewer2->addText("seg_source_point_cloud_image,total "+ to_string(seg_nums)+"parts", 10, 10, 1.0, 0.0, 0.0);
  viewer3->setBackgroundColor(255, 255, 255);//���ñ���ɫΪ��ɫ
  viewer3->addText("seg_target_point_cloud_image,total " + to_string(seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0);
  // �Ƚ���FPS�������õ��ָ���ƿ�����ĵ�
  vector<int> source_sampleIndex;
  vector<int> target_sampleIndex;
  ComputeFPS(source_point_cloud, seg_nums, source_sampleIndex);
  ComputeFPS(target_point_cloud, seg_nums, target_sampleIndex);
  // ����ָ��ĵ���
  pcl::PointCloud<pcl::PointXYZRGBA> PointCloud;
  vector<pcl::PointCloud<pcl::PointXYZRGBA>> NewSourceCloud(seg_nums, PointCloud);
  pcl::PointCloud<pcl::PointXYZRGBA> PointCloud1;
  vector<pcl::PointCloud<pcl::PointXYZRGBA>> NewTargetCloud(seg_nums, PointCloud1);
  // �任��ĵ���
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
  vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> TransformCloud(seg_nums*seg_nums, PointCloud2);
  // ����ת������
  Eigen::Matrix4f ransactransformation;
  Eigen::Matrix4f icptransformation;
  Eigen::Matrix4f finaltransformation;

  // ����TR��������
  vector<Eigen::Matrix4f> TR_vector(seg_nums*seg_nums);

  // ����RMSֵ
  double rms;
  double finalrms = 0.0;

  // ��RMSֵ��������������
  vector<double> rms_vector(seg_nums*seg_nums, -1);

  //����ѡȡ����������
  vector<vector<int>> NewSourceCloudIndex(seg_nums);
  vector<vector<int>> NewTargetCloudIndex(seg_nums);
  vector<vector<int>> PointRegionPair(seg_nums*seg_nums);
  int K = (int)(max(source_point_cloud->points.size(),target_point_cloud->points.size())/ seg_nums * K_factor); //ÿ�麬�еĵ�������
  cout << "ÿ��������Ƶ�����:"<<K << endl; //���������ʵ��overlap�����������

  //��,��,��,��,��,��,��,���,���,�������
  string colorname[16] = { "red","orange","yellow","green","cyan","blue","purple","darkRed","darkOrange","gold","oliveDrab","DarkTurquoise","DarkSlateBlue","DarkViolet","Pink", "Brown4"};
  double r[16] = {255,255,255,0,0,0,160,139,255,255,107,0,72,148,255,139};
  double g[16] = {0,165,255,255,255,0,32,0,127,215,142,206,61,0,192,35};
  double b[16] = {0,0,0,0,255,255,240,0,0,0,35,209,139,211,203,35};
  clock_t segment_start = clock();
#pragma omp parallel for
  for (int i = 0; i < seg_nums; i++)
  {
	  // ����
	  //cout << "��ʼ�ָ�Դ����" << endl;

	  // �ָ����
	  SegmentCloud(source_point_cloud, NewSourceCloud[i], NewSourceCloudIndex[i], source_sampleIndex[i], seg_nums, K);
	  //������������������������������������������������������������������������������������������������������������
	  //ʵ��һͼ
	/*   
	  target_sampleIndex[i] = source_sampleIndex[i];
	  NewTargetCloudIndex[i] = NewSourceCloudIndex[i];
	  NewTargetCloud[i] = NewSourceCloud[i];
	*/
	  //������������������������������������������������������������������������������������������������������������
	 SegmentCloud(target_point_cloud, NewTargetCloud[i], NewTargetCloudIndex[i], target_sampleIndex[i], seg_nums, K); 
  }
  clock_t segment_end = clock();
  cout << "�ָ��ҽ���,��ʱ:" << (double)(segment_end - segment_start) / (double)CLOCKS_PER_SEC << " s" << endl;
  for (int i = 0; i < seg_nums; i++)
  {

	  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> pointcloud_color_handler(255, 255, 255);//��ͬ�������ͬ����ɫ
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> part_source_pointcloud_color_handler(NewSourceCloud[i].makeShared(),r[i], g[i], b[i]);//��ͬ�������ͬ����ɫ
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> part_target_pointcloud_color_handler(NewTargetCloud[i].makeShared(), r[i], g[i], b[i]);//��ͬ�������ͬ����ɫ
	  viewer2->addPointCloud(NewSourceCloud[i].makeShared(), part_source_pointcloud_color_handler, "source_point_cloud" + to_string(i));
	  //viewer2->addText("part"+to_string(i)+" is "+ colorname[i], 10 + i * 50, 30, r[i]/255, g[i] / 255, b[i] / 255);
	  viewer2->addText("part" + to_string(i), 10 + i * 50, 30, r[i] / 255, g[i] / 255, b[i] / 255);
	  viewer3->addPointCloud(NewTargetCloud[i].makeShared(), part_target_pointcloud_color_handler, "target_point_cloud" + to_string(i));
	  //viewer3->addText("part"+to_string(i) + " is " + colorname[i], 10 + i * 50, 30, r[i] / 255, g[i] / 255, b[i] / 255);
	  viewer3->addText("part" + to_string(i), 10 + i * 50, 30, r[i] / 255, g[i] / 255, b[i] / 255);
	  // ����
	  //cout << "��" << i << "��ѭ������" << endl << endl;

  }
  //������������������������������������������������������������������������������������������������������������
  /*
  cout << "��Enter��������ѡ��" << endl;
  //����������
  int ch = 0;//�����������ֵ
  while (!viewer1->wasStopped())
  {
	  viewer1->spinOnce(100);   //
	  viewer2->spinOnce(100);   //
	  viewer3->spinOnce(100);
	  boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	  if (_kbhit()) {//����а������£���_kbhit()����������
		  ch = _getch();//ʹ��_getch()������ȡ���µļ�ֵ
	  }
	  if (ch == 13) { break; }//������Enterʱֹͣѭ����Enter���ļ�Ϊ13.
  }
  */

  //������������������������������������������������������������������������������������������������������������
   clock_t ransac_start = clock();
  //���Ʒֿ��ransac��׼
	//������������������������������������������������������������������������������������������������������������
   //ʵ��һ ��״ͼ
   double similar_block = 0;
   int similar_block_count = 0;
   double similar_block_average = 0;
   double unsimilar_block = 0;
   int unsimilar_block_count = 0;
   double unsimilar_block_average = 0;
#pragma omp parallel for
	   for (int i = 0; i < seg_nums; i++) {
		   for (int j = 0; j < seg_nums; j++) {
			   //ransac����Ĳ����������ü���ĵ����ܶȽ����������Ż���׼����͚G׼�ٶ�
			   //cout << "��ʼransac source���Ƶ�" << i << "���ֺ�target���Ƶ�" << j << "����,ʹ�õ��߳�Ϊ:" << omp_get_thread_num() << endl;
			   Ransac_registration(NewSourceCloud[i].makeShared(), NewTargetCloud[j].makeShared(), TR_vector[i*seg_nums + j], rms_vector[i*seg_nums + j]);
			   PointRegionPair[i*seg_nums + j] = { i,j };
			   cout << "��ransac source���Ƶ�" << i << "���ֺ�target���Ƶ�" << j << "����,rmsֵΪ"<<rms_vector[i*seg_nums + j]<<"��ʹ�õ��߳�Ϊ:"<< omp_get_thread_num() << endl;
			   if (i == j) {
				   similar_block = similar_block + rms_vector[i*seg_nums + j];
				   similar_block_count++;
			}
			   else
			   {
				   unsimilar_block = unsimilar_block + rms_vector[i*seg_nums + j];
				   unsimilar_block_count++;
			   }
		   }
	   }
  clock_t ransac_end = clock();
  cout << "ransac����,��ʱ:" << (double)(ransac_end - ransac_start) / (double)CLOCKS_PER_SEC << " s" << endl;
  similar_block_average = similar_block/ similar_block_count;
  unsimilar_block_average = unsimilar_block / unsimilar_block_count;
  cout << "similar_block_average:" << similar_block_average  << endl;
  cout << "unsimilar_block_average:" << unsimilar_block_average << endl;
  //������������������������������������������������������������������������������������������������������������

  //��ʾ��ѽ��
  /*
  //ֱ��ѡRMS��С�����
  double rms_min = DBL_MAX;
  int res_min_pair;
  // ����ѡȡ��С��rmsֵ ��Ӧ��TR����
  for (int k = 0; k < TR_vector.size(); k++)
  {
	  if (rms_vector[k] < rms_min)
	  {
		  rms_min = rms_vector[k];
		  ransactransformation = TR_vector[k];
		  res_min_pair = k;
	  }
  }
 */
  //ѡ�����RT������RMS��С�����
  int res_min_pair = ComputerRelativeTR_Threshold(rms_vector, TR_vector, source_point_cloud_density,PointRegionPair);
  double rms_min = rms_vector[res_min_pair];
  ransactransformation = TR_vector[res_min_pair];
  //�����ȡ
  int pc1;
  int pc2;
  vector<int> pcpair;
  //cout << "res_min_pair��" << res_min_pair << endl;
  pcpair = PointRegionPair[res_min_pair];
  //cout << "pcpair.size��" << pcpair.size() << endl;
  pc1 = pcpair[0];
  pc2 = pcpair[1];


  // ����С��RMSֵ���
  cout << "��С��RMSֵ��" << rms_min << endl;
  // ����С��rms��Ӧ��TR����任����
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pair1ransacCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pair1icpCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr OutCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  cout << "pc1��" << pc1 << endl;
  cout << "pc2��" << pc2 << endl;

  //pcl::transformPointCloud(*source_point_cloud, *OutCloud, ransactransformation);
  pcl::transformPointCloud(*NewSourceCloud[pc1].makeShared(), *pair1ransacCloud, ransactransformation);
  method_icp(pair1ransacCloud, NewTargetCloud[pc2].makeShared(), pair1icpCloud, icptransformation);
  finaltransformation =  icptransformation *ransactransformation;
  pcl::transformPointCloud(*source_point_cloud, *OutCloud, finaltransformation);
	  //���յ�����׼�����RMSֵ
  /*
  ComputeRMS(OutCloud, target_point_cloud, finalrms);
  cout << "���յ�����׼RMSֵ��" << finalrms << endl;
 */ 
 //TRMS��С������	
  boost::shared_ptr<vector<int>> index_ptr1 = boost::make_shared<vector<int>>(NewSourceCloudIndex[pc1]);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc1PairRegion(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::ExtractIndices<pcl::PointXYZRGBA> pair1Indice;
  pair1Indice.setInputCloud(OutCloud);
  pair1Indice.setIndices(index_ptr1);
  pair1Indice.setNegative(false);                    //�����Ϊtrue, ������ȡָ��index֮��ĵ���
  pair1Indice.filter(*pc1PairRegion);


  boost::shared_ptr<vector<int>> index_ptr2 = boost::make_shared<vector<int>>(NewTargetCloudIndex[pc2]);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc2PairRegion(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::ExtractIndices<pcl::PointXYZRGBA> pair2Indice;
  pair2Indice.setInputCloud(target_point_cloud);
  pair2Indice.setIndices(index_ptr2);
  pair2Indice.setNegative(false);                    //�����Ϊtrue, ������ȡָ��index֮��ĵ���
  pair2Indice.filter(*pc2PairRegion);

  //-------------------------------------------------------------------------------------------------------------------------------------
 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer4(new pcl::visualization::PCLVisualizer("ѡȡ�ĵ�������"));
 viewer4->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
 viewer4->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
 viewer4->setBackgroundColor(255, 255, 255, v1);//���ñ���ɫΪ��ɫ
 viewer4->setBackgroundColor(255, 255, 255, v2);//���ñ���ɫΪ��ɫ
 viewer4->addPointCloud(source_point_cloud, "source_point_cloud", v1);
 viewer4->addPointCloud(target_point_cloud, "target_point_cloud", v2);
 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> sourcePairRegion_cloud_color_handler(NewSourceCloud[pc1].makeShared(), r[pc1], g[pc1], b[pc1]);//��ɫ
 viewer4->addPointCloud(NewSourceCloud[pc1].makeShared(), sourcePairRegion_cloud_color_handler, "sourcePairRegion",v1);
 viewer4->addText("seg_source_point_cloud_image,total " + to_string(seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0, "v1 text", v1);
 viewer4->addText("select part" + to_string(pc1), 10 , 30, r[pc1] / 255, g[pc1] / 255, b[pc1] / 255, "v1 select text", v1);
 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> targetPairRegion_cloud_color_handler(NewTargetCloud[pc2].makeShared(), r[pc2], g[pc2], b[pc2]);//��ɫ
 viewer4->addPointCloud(NewTargetCloud[pc2].makeShared(), targetPairRegion_cloud_color_handler, "targetPairRegion",v2);
 viewer4->addText("seg_target_point_cloud_image,total " + to_string(seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0, "v2 text", v2);
 viewer4->addText("select part" + to_string(pc2), 10 , 30, r[pc2] / 255, g[pc2] / 255, b[pc2] / 255, "v2 select text", v2);
 viewer4->spinOnce(100);

  // ��ʾ���Ʋ���
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("���ǵ��㷨��׼��Դ��Ŀ����ƿ��ӻ�"));
  viewer->setBackgroundColor(255, 255, 255);//���ñ���ɫΪ��ɫ
  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> source_cloud_color_handler(OutCloud, 0, 255, 0);//������
  viewer->addText("source cloud select part" + to_string(pc1), 10, 10, r[pc1] / 255, g[pc1] / 255, b[pc1] / 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> best_sourcePairRegion_cloud_color_handler(NewSourceCloud[pc1].makeShared(), r[pc1], g[pc1], b[pc1]);
  viewer->addPointCloud(OutCloud,  "source_point_cloud");
  viewer->addPointCloud(pc1PairRegion, best_sourcePairRegion_cloud_color_handler, "sourcePairRegion");
  viewer->addText("target cloud select part" + to_string(pc2), 10, 30, r[pc2] / 255, g[pc2] / 255, b[pc2] / 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> best_targetPairRegion_cloud_color_handler(NewTargetCloud[pc2].makeShared(), r[pc2], g[pc2], b[pc2]);
  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> targetPairRegion_cloud_color_handler(pc2PairRegion, 250, 0, 0);//���
  viewer->addPointCloud(target_point_cloud, "target_point_cloud");
  viewer->addPointCloud(pc2PairRegion, best_targetPairRegion_cloud_color_handler, "targetPairRegion_cloud");
  //������������������������������������������������������������������������������������������������������������
  //������ת����
  cout << pc1name << endl;
  int begin = pc2name.find_last_of('\\');
  string origin_transform_matrix_filepath = pc1name.substr(0, pc1name.size() - 4) + "_part" + to_string(pc1) + "_" + pc2name.substr(begin+1, pc2name.size() - 4) + "_part" + to_string(pc2) + ".txt";
 // string origin_transform_matrix_filepath = pc1name.substr(0, pc1name.size() - 4) + "_part" + to_string(pc1) + "_" + pc2name.substr(0, pc2name.size() - 4) + "_part" + to_string(pc2) + ".txt";
  cout << origin_transform_matrix_filepath << endl;
  std::ofstream  origin;
  origin.open(origin_transform_matrix_filepath, std::ios::app);//���ļ�ĩβ׷��д��
  origin << finaltransformation << std::endl;//ÿ��д��һ�������Ժ���
  origin.close();
  //������������������������������������������������������������������������������������������������������������
  cout << "�粻���⣬��Enter�ֶ���������ѡ��" << endl;
  //����������
  int ch = 0;//�����������ֵ
  while (1)
  {
	  //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	  viewer->spinOnce(100);
	  viewer1->spinOnce(100);   //
	  viewer2->spinOnce(100);   //
	  viewer3->spinOnce(100);
	  viewer4->spinOnce(100);
	  if (_kbhit()) {//����а������£���_kbhit()����������
		  ch = _getch();//ʹ��_getch()������ȡ���µļ�ֵ
		  cout << ch << endl;
	  }
	  if (ch == 13) { break; }//������Enterʱֹͣѭ����Enter���ļ�Ϊ13.
  }
  //������������������������������������������������������������������������������������������������������������
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer5(new pcl::visualization::PCLVisualizer("�ֶ�ѡȡ�ĵ�������"));
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer6(new pcl::visualization::PCLVisualizer("���ǵ��㷨�ֶ�ѡ����׼��Դ��Ŀ����ƿ��ӻ�"));
  while (1) {
	  viewer5->close();
	  viewer6->close();
	  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer5(new pcl::visualization::PCLVisualizer("�ֶ�ѡȡ�ĵ�������"));
	  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer6(new pcl::visualization::PCLVisualizer("���ǵ��㷨�ֶ�ѡ����׼��Դ��Ŀ����ƿ��ӻ�"));
	  //������������������������������������������������������������������������������������������������������������
	  //ѡȡ���Ʒֿ����ʾ
	  string sourcepart;
	  string targetpart;
	  cout << "Input source point cloud part number from 0 to" << seg_nums - 1 << endl;
	  getline(cin, sourcepart);
	  cout << "You select source point cloud part" << sourcepart << "\n";
	  int sourcepartnum = stoi(sourcepart);
	  while (sourcepartnum > seg_nums - 1) {
		  cout << "Wrong number! Input source point cloud part number from 0 to" << seg_nums - 1 << endl;
		  getline(cin, sourcepart);
		  sourcepartnum = stoi(sourcepart);
		  cout << "You select source point cloud part" << sourcepart << "\n";
	  }
	  cout << "Input target point cloud part number from 0 to" << seg_nums - 1 << endl;
	  getline(cin, targetpart);
	  cout << "You select target point cloud part" << targetpart << "\n";
	  int targetpartnum = stoi(targetpart);
	  while (targetpartnum > seg_nums - 1) {
		  cout << "Wrong number! Input target point cloud part number from 0 to" << seg_nums - 1 << endl;
		  getline(cin, targetpart);
		  targetpartnum = stoi(targetpart);
		  cout << "You select target point cloud part" << targetpart << "\n";
	  }

	  //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer5(new pcl::visualization::PCLVisualizer("�ֶ�ѡȡ�ĵ�������"));
	  viewer5->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	  viewer5->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	  viewer5->setBackgroundColor(255, 255, 255, v1);//���ñ���ɫΪ��ɫ
	  viewer5->setBackgroundColor(255, 255, 255, v2);//���ñ���ɫΪ��ɫ
	  viewer5->addPointCloud(source_point_cloud, "source_point_cloud", v1);
	  viewer5->addPointCloud(target_point_cloud, "target_point_cloud", v2);
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> manual_sourcePairRegion_cloud_color_handler(NewSourceCloud[sourcepartnum].makeShared(), r[sourcepartnum], g[sourcepartnum], b[sourcepartnum]);//��ɫ
	  viewer5->addPointCloud(NewSourceCloud[sourcepartnum].makeShared(), manual_sourcePairRegion_cloud_color_handler, "sourcePairRegion", v1);
	  viewer5->addText("seg_source_point_cloud_image,total " + to_string(seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0, "v1 text", v1);
	  viewer5->addText("select part" + sourcepart, 10, 30, r[sourcepartnum] / 255, g[sourcepartnum] / 255, b[sourcepartnum] / 255, "v1 select text", v1);
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> manual_targetPairRegion_cloud_color_handler(NewTargetCloud[targetpartnum].makeShared(), r[targetpartnum], g[targetpartnum], b[targetpartnum]);//��ɫ
	  viewer5->addPointCloud(NewTargetCloud[targetpartnum].makeShared(), manual_targetPairRegion_cloud_color_handler, "targetPairRegion", v2);
	  viewer5->addText("seg_target_point_cloud_image,total " + to_string(seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0, "v2 text", v2);
	  viewer5->addText("select part" + targetpart, 10, 30, r[targetpartnum] / 255, g[targetpartnum] / 255, b[targetpartnum] / 255, "v2 select text", v2);
	  viewer5->spinOnce(100);

	  Eigen::Matrix4f manual_finaltransformation;
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr manual_pair1ransacCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr manual_pair1icpCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr manual_OutCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  //pcl::transformPointCloud(*source_point_cloud, *OutCloud, ransactransformation);
	  method_sac_icp(NewSourceCloud[sourcepartnum].makeShared(), NewTargetCloud[targetpartnum].makeShared(), manual_finaltransformation);
	  //pcl::transformPointCloud(*NewSourceCloud[sourcepartnum].makeShared(), *manual_pair1ransacCloud, manual_ransactransformation);
	  //method_icp(pair1ransacCloud, NewTargetCloud[targetpartnum].makeShared(), manual_pair1icpCloud, manual_icptransformation);
	  //manual_finaltransformation = manual_icptransformation * manual_ransactransformation;
	  pcl::transformPointCloud(*source_point_cloud, *manual_OutCloud, manual_finaltransformation);
	  //���յ�����׼�����RMSֵ
	/*
	ComputeRMS(OutCloud, target_point_cloud, finalrms);
	cout << "���յ�����׼RMSֵ��" << finalrms << endl;
	*/
	//TRMS��С������	
	  boost::shared_ptr<vector<int>> index_manual_ptr1 = boost::make_shared<vector<int>>(NewSourceCloudIndex[sourcepartnum]);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr manual_PairRegion1(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::ExtractIndices<pcl::PointXYZRGBA> manual_pair1Indice;
	  pair1Indice.setInputCloud(manual_OutCloud);
	  pair1Indice.setIndices(index_manual_ptr1);
	  pair1Indice.setNegative(false);                    //�����Ϊtrue, ������ȡָ��index֮��ĵ���
	  pair1Indice.filter(*manual_PairRegion1);


	  boost::shared_ptr<vector<int>> index_manual_ptr2 = boost::make_shared<vector<int>>(NewTargetCloudIndex[targetpartnum]);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr manual_PairRegion2(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::ExtractIndices<pcl::PointXYZRGBA> manual_pair2Indice;
	  pair2Indice.setInputCloud(target_point_cloud);
	  pair2Indice.setIndices(index_manual_ptr2);
	  pair2Indice.setNegative(false);                    //�����Ϊtrue, ������ȡָ��index֮��ĵ���
	  pair2Indice.filter(*manual_PairRegion2);

	  // ��ʾ���Ʋ���
	 //boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer6(new pcl::visualization::PCLVisualizer("���ǵ��㷨�ֶ�ѡ����׼��Դ��Ŀ����ƿ��ӻ�"));
	  viewer6->setBackgroundColor(255, 255, 255);//���ñ���ɫΪ��ɫ
	  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> source_cloud_color_handler(OutCloud, 0, 255, 0);//������
	  viewer6->addText("source cloud select part" + to_string(sourcepartnum), 10, 10, r[sourcepartnum] / 255, g[sourcepartnum] / 255, b[sourcepartnum] / 255);
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> manual_sourcePairRegion_resultcloud_color_handler(NewSourceCloud[sourcepartnum].makeShared(), r[sourcepartnum], g[sourcepartnum], b[sourcepartnum]);
	  viewer6->addPointCloud(manual_OutCloud, "source_point_cloud");
	  viewer6->addPointCloud(manual_PairRegion1, manual_sourcePairRegion_resultcloud_color_handler, "manual_PairRegion1_cloud");
	  viewer6->addText("target cloud select part" + to_string(targetpartnum), 10, 30, r[targetpartnum] / 255, g[targetpartnum] / 255, b[targetpartnum] / 255);
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> manual_targetPairRegion_resultcloud_color_handler(NewTargetCloud[targetpartnum].makeShared(), r[targetpartnum], g[targetpartnum], b[targetpartnum]);
	  //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> targetPairRegion_cloud_color_handler(targetpartnum PairRegion, 250, 0, 0);//���
	  viewer6->addPointCloud(target_point_cloud, "target_point_cloud");
	  viewer6->addPointCloud(manual_PairRegion2, manual_targetPairRegion_resultcloud_color_handler, "manual_PairRegion2_cloud");
 
	  //������������������������������������������������������������������������������������������������������������
	  //������ת����
	  string transform_matrix_filepath = pc1name.substr(0,pc1name.size()-4)+"_part"+ sourcepart+"_"+ pc2name.substr(begin + 1, pc2name.size() - 4) + "_part" + targetpart+ ".txt";
	  std::ofstream fout;
	  fout.open(transform_matrix_filepath, std::ios::app);//���ļ�ĩβ׷��д��
	  fout << manual_finaltransformation << std::endl;//ÿ��д��һ�������Ժ���
	  fout.close();
	  //������������������������������������������������������������������������������������������������������������
	  int ch = 0;//�����������ֵ
	  cout << "�粻���⣬��Enter�ֶ���������ѡ��" << endl;
	  while (!viewer->wasStopped())
	  {
		  //cout << "�粻���⣬��Enter�ֶ���������ѡ��" << endl;
		  viewer->spinOnce(100);
		  viewer1->spinOnce(100);   //
		  viewer2->spinOnce(100);   //
		  viewer3->spinOnce(100);
		  viewer4->spinOnce(100);
		  viewer5->spinOnce(100);
		  viewer6->spinOnce(100);
		  //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
		  //cout << ch << endl;
		  if (_kbhit()) {//����а������£���_kbhit()����������
			  ch = _getch();//ʹ��_getch()������ȡ���µļ�ֵ
		  }
		  if (ch == 13) { break; }//������Enterʱֹͣѭ����Enter���ļ�Ϊ13.
	  }
  }
 return (0);
}
