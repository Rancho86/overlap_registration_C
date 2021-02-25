#include<conio.h>
#include<vector>
#include<iostream>
#include<string>
#include<ctime>
#include<algorithm>
#include<math.h>
#include<omp.h> 
//pointcloud_read
#include <pcl/PolygonMesh.h>
#include <pcl/io/io.h>
#include<pcl/io/obj_io.h>
#include<pcl/io/ply_io.h>
#include<pcl/io/pcd_io.h>
#include <pcl/io/vtk_lib_io.h>
#include<pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>	
//Downsample
#include <pcl/filters/random_sample.h> 
#include<pcl/filters/extract_indices.h>
#include<pcl/filters/voxel_grid.h>
#include<pcl/kdtree/kdtree_flann.h>
#include <pcl/keypoints/uniform_sampling.h>
//Ransac
#include<pcl/sample_consensus/sac_model_registration.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ia_ransac.h>
//Feature
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh_omp.h>
using namespace std;


void printUsage(const char* progName)
{
	cout << "First enter the two point cloud names, and then add the options that need to be set" << endl;
	cout << "-r Add registration type, 0 means ¡®whole to whole¡¯, 1 means ¡®part to whole¡¯, 2 means ¡®part to part¡¯, 3 means ¡¯unknown¡®" << endl;
	cout << "-o Add overlap rate,the default is 0.3" << endl;
	cout << "-f Add downsample factor,the default is 0.2" << endl;
	cout << "-p Add the number of blocks,the default is calculated by overlap rate" << endl;
	cout << "For example£º"<<progName<<" source_point_cloud.pcd/obj/ply target_point_cloud.pcd/obj/ply -r 1 -o 0.6 -f 1 -p 10" << endl;
	cout << "Designed by Rancho and Yinhui Wang" << endl;
}

//Read point cloud file//
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
		//cout << "start reading source_point_cloud" <<  endl;
		pcl::io::loadPolygonFile(argv[1], mesh1);
		pcl::fromPCLPointCloud2(mesh1.cloud, *source_point_cloud);

	}
		//pcl::io::loadOBJFile(argv[1], *source_point_cloud); //This method of reading is extremely slow
	if (pc1ext.compare("pcd") == 0)
		pcl::io::loadPCDFile(argv[1], *source_point_cloud);
	if (pc2ext.compare("ply") == 0)
		pcl::io::loadPLYFile(argv[2], *target_point_cloud);
	if (pc2ext.compare("obj") == 0)		
	{
		//cout << "start reading target_point_cloud" << endl;
		pcl::io::loadPolygonFile(argv[2], mesh2);
		pcl::fromPCLPointCloud2(mesh2.cloud, *target_point_cloud);
	}
		//pcl::io::loadOBJFile(argv[2], *target_point_cloud); //This method of reading is extremely slow
	if (pc2ext.compare("pcd") == 0)
		pcl::io::loadPCDFile(argv[2], *target_point_cloud);
	
	//Remove nan points
	std::vector<int> indices_source_nan_indice;
	std::vector<int> indices_target_nan_indice;
	pcl::removeNaNFromPointCloud(*source_point_cloud, *source_point_cloud, indices_source_nan_indice);
	pcl::removeNaNFromPointCloud(*target_point_cloud, *target_point_cloud, indices_target_nan_indice);

	clock_t read_pc_end = clock();
	cout << "read pointcloud time: " << (double)(read_pc_end - read_pc_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	cout << "source_point_cloud size : " << source_point_cloud->points.size() << endl;
	cout << "target_point_cloud size : " << target_point_cloud->points.size() << endl;
}

//save as ply
void savePointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, std::string outpath)
{
	std::cerr << "save path is :" << outpath << endl;
	//½«string±£´æÂ·¾¶×ªÎªchar*
	char *path = new char[outpath.size() + 1];
	strcpy(path, outpath.c_str());
	//std::cerr << "Path is : " << path << " ." << std::endl;

	//Ð´³öµãÔÆÍ¼
	pcl::PLYWriter writer;
	writer.write(path, *cloud, true);
	//std::cerr << "PointCloud has : " << cloud->width * cloud->height << " data points." << std::endl;
}


//Calculate point cloud resolution
double
computeCloudResolution(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
	double res = 0.0;
	int n_points = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> sqr_distances(2);
	pcl::search::KdTree<pcl::PointXYZRGBA> tree;
	tree.setInputCloud(cloud);

	for (std::size_t i = 0; i < cloud->size(); ++i)
	{
		if (!std::isfinite((*cloud)[i].x))
		{
			continue;
		}
		//Considering the second neighbor since the first is the point itself.
		nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
		if (nres == 2)
		{
			res += sqrt(sqr_distances[1]);
			++n_points;
		}
	}
	if (n_points != 0)
	{
		res /= n_points;
	}
	return res;
}

void compute_error(const Eigen::Matrix4f TR_registration, const Eigen::Matrix4f TR_true, double &Rerror, double &Terror) {
	Eigen::Matrix4f TR_error;
	TR_error = TR_registration * TR_true.inverse();
	cout << TR_error << endl;
	Eigen::Matrix3f TR_Diff_R;
	TR_Diff_R = TR_error.block<3, 3>(0, 0);
	cout << TR_Diff_R << endl;
	Rerror = fabs(acos((TR_Diff_R.trace() - 1) / 2)) * 180 / 3.14;
	Terror = (fabs(TR_error(0, 3)) + fabs(TR_error(1, 3)) + fabs(TR_error(2, 3))) / 3;
}

//Random Downsample
void DownSample(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud,int count)
{
	pcl::RandomSample<pcl::PointXYZRGBA> sor;
	sor.setInputCloud(cloud);
	sor.setSample(count);
	sor.filter(*cloud);
}


//FPS
void ComputeFPS(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, const int nums, vector<int> &SampleIndex)
{
	int sample_nums = nums;
	int cloud_points_size = cloud->points.size();
	vector<int> rest_cloud_pointsIndex;
	vector<double> rest2sample_dist(cloud_points_size);
	for (int i = 0; i < cloud_points_size; i++)
		rest_cloud_pointsIndex.push_back(i);

	double max_dist = DBL_MIN;         // Record the maximum distance, use when picking the second point
	double max_dist_;                 // Record the maximum distance, use when picking the second and subsequent points
	int farthest_point = -1;          // Record farthest point
	double length_dist=DBL_MAX;               // The distance between points 
	int sample_inteIndex = -1;           // Relative segment position in sample point cloud
	int sample_inteIndex_value = -1;     // The relative position in the sampled point cloud corresponds to the index of the real point cloud
	int inteIndex_value = -1;         // The remaining point cloud index corresponds to the index of the real point cloud
	srand(time(NULL));                // Initialize random seed with system time

	clock_t  sample_start, sample_stop;
	sample_start = clock();
	int first_select = rand() % (rest_cloud_pointsIndex.size() + 1);
	int first_select_value = rest_cloud_pointsIndex[first_select];
	SampleIndex.push_back(rest_cloud_pointsIndex[first_select]);
	vector<int>::iterator iter = rest_cloud_pointsIndex.begin() + first_select;
	rest_cloud_pointsIndex.erase(iter);
	if (nums > 1) {
		for (int j = 0; j < rest_cloud_pointsIndex.size(); j++)
		{
			inteIndex_value = rest_cloud_pointsIndex[j];
			// Calculate distance
			length_dist = pow(abs(cloud->points[inteIndex_value].x - cloud->points[first_select_value].x), 2) +
				pow(abs(cloud->points[inteIndex_value].y - cloud->points[first_select_value].y), 2) +
				pow(abs(cloud->points[inteIndex_value].z - cloud->points[first_select_value].z), 2);

			rest2sample_dist[j] = length_dist;              // Save the minimum distance from each point in the point cloud to be sampled to the sample point cloud
			if (length_dist > max_dist)
			{
				max_dist = length_dist;
				farthest_point = j;
			}
		}
		// Save the index of the second furthest point in the partition
		SampleIndex.push_back(rest_cloud_pointsIndex[farthest_point]);
		// Delete the point from the remaining point cloud
		iter = rest_cloud_pointsIndex.begin() + farthest_point;
		rest_cloud_pointsIndex.erase(iter);

		if (nums > 2) {
			while (SampleIndex.size() < sample_nums)
			{
				max_dist_ = DBL_MIN;                

				// Traverse the remaining point cloud

				for (int j = 0; j < rest_cloud_pointsIndex.size(); j++)
				{
					length_dist = DBL_MAX;
					inteIndex_value = rest_cloud_pointsIndex[j];
					// Calculate the distance between the remaining point cloud and the newly selected point, and compare the distance with the previously selected point to take the minimum value, and update the distance matrix
					// Calculate distance
					for (int i = 0; i < SampleIndex.size(); i++) {
						sample_inteIndex_value = SampleIndex[i];
						double length_dist_a = pow(abs(cloud->points[inteIndex_value].x - cloud->points[sample_inteIndex_value].x), 2) +
							pow(abs(cloud->points[inteIndex_value].y - cloud->points[sample_inteIndex_value].y), 2) +
							pow(abs(cloud->points[inteIndex_value].z - cloud->points[sample_inteIndex_value].z), 2);
						length_dist = min(length_dist, length_dist_a);
					}

					// Compare distance and update matrix
					rest2sample_dist[j] = length_dist;

					if (rest2sample_dist[j] > max_dist_)
					{
						farthest_point = j;
						max_dist_ = rest2sample_dist[j];
					}
				}
				// Save the selected farthest point to the split point cloud
				SampleIndex.push_back(rest_cloud_pointsIndex[farthest_point]);
				// Delete the point from the remaining point cloud
				iter = rest_cloud_pointsIndex.begin() + farthest_point;
				rest_cloud_pointsIndex.erase(iter);

			}
		}
	}
	sample_stop = clock();

	cout << "FPS sampling completed£¡" << endl;
	cout << "Number of seed points£º" << SampleIndex.size() << endl;
	cout << "FPS Time£º" << (double)(sample_stop - sample_start) / (double)CLOCKS_PER_SEC << "s" << endl;
}

// ---------------------------------------------------------------------------
// -----------------------------Point cloud block program ---------------------------------
// ---------Use the point sampled by FPS as the center point, and use the KD tree to obtain the point cloud block --------------------
// ---------------------------------------------------------------------------
void SegmentCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBA> &newcloud, vector<int> &pointIdxNKNSearch, const int index, const int seg_nums, const int K,const int sample_num)
{
	clock_t  seg_start, seg_stop;
	seg_start = clock();
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZRGBA searchPoint = cloud->points[index];

	vector<float> pointNKNSquaredDistance(K);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_ptr(new pcl::PointCloud < pcl::PointXYZRGBA>);
	pcl::ExtractIndices<pcl::PointXYZRGBA> extract;

	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		boost::shared_ptr<vector<int>> pointIdxRadiusSearch_ptr = boost::make_shared<vector<int>>(pointIdxNKNSearch);
		// Extract point cloud based on index
		extract.setInputCloud(cloud);
		extract.setIndices(pointIdxRadiusSearch_ptr);
		extract.setNegative(false);                    //If set to true, point clouds outside the specified index can be extracted
		extract.filter(*cloud_ptr);
		//newcloud = *cloud_ptr;
	}

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr newcloud_s(new pcl::PointCloud < pcl::PointXYZRGBA>);
	newcloud_s = newcloud.makeShared();
	if (cloud_ptr->points.size()> sample_num)
		DownSample(cloud_ptr, sample_num);
	newcloud = *cloud_ptr;
	seg_stop = clock();
	//cout << "Time to obtain the point cloud£º" << seg_stop - seg_start << "s" << endl;
}

// ---------------------------------------------------------------------------------------
// ------------------------------- Use sac-ia to calculate similarity and transformation matrix --------------------------------------------
// ---------------------------------------------------------------------------------------
void Ransac_registration(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud,
	Eigen::Matrix4f &transformation, double &score)
{	
	clock_t  ransac_start, ransac_end;
	clock_t  density_start, density_end;
	clock_t  downsample_start, downsample_end;
	clock_t  normal_start, normal_end;
	clock_t  fpfh_start, fpfh_end;
	clock_t  sac_start, sac_end;

	//Calculate point cloud resolution	
	ransac_start = clock();
	density_start = clock();
	double sourcepoint_leafsize;
	sourcepoint_leafsize = computeCloudResolution(source_point_cloud)*5;
	double targetpoint_leafsize;
	targetpoint_leafsize = computeCloudResolution(target_point_cloud)*5;
	density_end = clock();
	//Downsample	
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
	//cout << "Downsample time:" << (double)(downsample_end - downsample_start) / (double)CLOCKS_PER_SEC << " s" << endl;

	//Calculate surface normal
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
	//cout << "Calculate surface normal time:" << (double)(normal_end - normal_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	
	//calculate FPFH
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
	//cout << "fpfh time:" << (double)(fpfh_end - fpfh_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	
	//sac registration
	sac_start = clock();
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::FPFHSignature33> scia;
	scia.setInputSource(SourceCloud);
	scia.setInputTarget(TargetCloud);
	scia.setSourceFeatures(fpfhs_src);
	scia.setTargetFeatures(fpfhs_tgt);
	scia.setMaxCorrespondenceDistance(sourcepoint_leafsize * 2 * sourcepoint_leafsize * 2);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
	scia.align(*sac_result);
	//cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	transformation = scia.getFinalTransformation();
    //cout << transformation << endl;
	sac_end = clock();
	//cout << "sac time:" << (double)(sac_end - sac_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	ransac_end = clock();
	//cout << "ransac time:" << (double)(ransac_end - ransac_start) / (double)CLOCKS_PER_SEC << " s" << endl;
	if (scia.hasConverged())
		score = scia.getFitnessScore();
}

void method_sac_icp(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud, const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud,  Eigen::Matrix4f &registration_matrix)
{
	double sourcepoint_leafsize;
	sourcepoint_leafsize = computeCloudResolution(source_point_cloud);//*5
	double targetpoint_leafsize;
	targetpoint_leafsize = computeCloudResolution(target_point_cloud);
	vector<int> indices_src; //Save the index of the removed point
	pcl::removeNaNFromPointCloud(*source_point_cloud, *source_point_cloud, indices_src);
	//cout << "remove *source_point_cloud nan" << endl;
	//Downsample
	pcl::VoxelGrid<pcl::PointXYZRGBA> voxel_grid;
	voxel_grid.setLeafSize(sourcepoint_leafsize*2, sourcepoint_leafsize * 2, sourcepoint_leafsize * 2);
	voxel_grid.setInputCloud(source_point_cloud);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_src(new pcl::PointCloud<pcl::PointXYZRGBA>);
	voxel_grid.filter(*cloud_src);
	//cout << "down size *source_point_cloud from " << source_point_cloud->size() << "to" << cloud_src->size() << endl;
	//Calculate surface normal
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

	//calculate FPFH
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
	//cout << "compute *cloud_tgt fpfh" << endl;

	//SAC registration
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZRGBA, pcl::PointXYZRGBA, pcl::FPFHSignature33> scia;
	scia.setInputSource(cloud_src);
	scia.setInputTarget(cloud_tgt);
	scia.setSourceFeatures(fpfhs_src);
	scia.setTargetFeatures(fpfhs_tgt);
	scia.setMaxCorrespondenceDistance(sourcepoint_leafsize * 2* sourcepoint_leafsize * 2);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
	scia.align(*sac_result);
	cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	Eigen::Matrix4f sac_trans;
	sac_trans = scia.getFinalTransformation();
	cout << sac_trans << endl;
	clock_t sac_time = clock();

	//icp registration
	pcl::IterativeClosestPoint<pcl::PointXYZRGBA, pcl::PointXYZRGBA> icp;
	//icp.setMaxCorrespondenceDistance(0.04);
	icp.setMaxCorrespondenceDistance(sourcepoint_leafsize * 5);
	icp.setMaximumIterations(50);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(0.2);
	icp.setInputSource(cloud_src);
	icp.setInputTarget(cloud_tgt);
	pcl::PointCloud<pcl::PointXYZRGBA> Final;
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZRGBA>);
	//icp.align(Final);
	icp.align(*icp_result,sac_trans);
	cout << "icp has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << endl;
	cout << icp.getFinalTransformation() << endl;
	Eigen::Matrix4f icp_trans= icp.getFinalTransformation();
	registration_matrix = icp_trans;
}


// -----------------------------------------------------------------------------------------------
// ---------------------------TAM module------------------------------------------------
// ------------------Select the candidate block pair with the highest similarity------- 
// -------------------calculate their approximate matrix-----------------------------------
// -------------------output the appropriate block pair-----------------------------------
// -----------------------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------------------

int ComputerRelativeTR_Threshold(const vector<double> rms_vector, const vector<Eigen::Matrix4f> TR_vector, double pointdis,vector<vector<int>>regionpair)
{
	vector<double> rms_copy;

	for (int i = 0; i < rms_vector.size(); i++)
		rms_copy.push_back(rms_vector[i]);

	vector<int> rms_copy_index;

	for (int j = 0; j < rms_vector.size(); j++)
		rms_copy_index.push_back(j);

	vector<int> rms_min_index;              
	Eigen::Matrix4f TR_Diff;                
	Eigen::Matrix3f TR_Diff_3;
	int candidate_num = min(int(ceil(rms_vector.size()*0.5)), 10);
	vector<vector<int>> similartPair(candidate_num);
	vector<double> count_vector;                 


	for (int i = 0; i < candidate_num; i++)        
	{
		double rms_Min = DBL_MAX;          
		int Min_Index = 0;
		for (int j = 0; j < rms_copy.size(); j++)     
		{
			if (rms_copy[j] < rms_Min)
			{
				rms_Min = rms_copy[j];
				Min_Index = j;
			}
		}

		rms_min_index.push_back(rms_copy_index[Min_Index]);

		// Delete the current minimum
		vector<double>::iterator iter1 = rms_copy.begin() + Min_Index;
		rms_copy.erase(iter1);
		// Delete the index corresponding to the current minimum value
		vector<int>::iterator iter2 = rms_copy_index.begin() + Min_Index;
		rms_copy_index.erase(iter2);

	}

	// Define the variable that records the mean of the rotation difference
	double rotation_dif = 0.0;
	// Define the variable that records the mean of the translation difference
	double translation_dif = 0.0;

	// Define the rotation threshold
	double rotation_threshold = 20;
	cout << "rotation_threshold:" << rotation_threshold << endl;
	// Define the translation threshold
	double translation_threshold = pointdis * 50;
	cout << "translation_threshold:" << translation_threshold << endl;
	int total_count = 0;
	for (int m = 0; m < candidate_num; m++)
	{
		//Record the number of approximate matrices that meet the conditions
		int count = 0;
		for (int n = 0; n < candidate_num; n++)
		{
			if (n != m)
			{
				//Calculate the transformation matrix between two registration matrices
				TR_Diff = TR_vector[rms_min_index[m]].inverse() * (TR_vector[rms_min_index[n]]);
				Eigen::Matrix3f TR_Diff_R;
				TR_Diff_R = TR_Diff.block<3, 3>(0, 0);
				//Rotation matrix to find the rotation angle
				rotation_dif=fabs(acos((TR_Diff_R.trace() - 1) / 2)) * 180 / 3.14;
				cout << "Rotation angle difference of RT matrix with matrx " << n << " is£º" << rotation_dif << endl;
				// Calculate the mean of the translation difference of the TR matrix
				translation_dif= (fabs(TR_Diff(0, 3)) + fabs(TR_Diff(1, 3)) + fabs(TR_Diff(2, 3))) / 3;

				cout << "Transform difference of RT matrix with matrx " << n << " is£º" << translation_dif << endl;

			

				// ½«ÉÏÊö¾ùÖµÓë¶ÔÓ¦ãÐÖµ½øÐÐ±È½Ï
				if ((rotation_dif < rotation_threshold) && (translation_dif < translation_threshold))
				{
						count += 1;
						similartPair[m].push_back(n);
				}
					
			}
		}
		// ²âÊÔ
		// ÏÔÊ¾ÆärmsÖµ
	    cout << "The similarity between block pair"<< regionpair[rms_min_index[m]][0]<<" and "<< regionpair[rms_min_index[m]][1] <<"is" << rms_vector[rms_min_index[m]] << endl;
		cout << "The count of approximate matrices£º" << count << endl;
		total_count = total_count + count;
		count_vector.push_back(count);
	}

	int count_average = total_count/ candidate_num;         
	int count_selected = 0;
	int count_selected_index = 0;
	for (int k = 0; k < candidate_num; k++)
	{
		if (count_vector[k] > count_average)
		{
			count_selected = count_vector[k];
			count_selected_index = k;
			break;
		}
	}
	cout << "The average of the approximate matrix£º" << count_average << endl;
	cout << "Number of approximate matrices selected£º" << count_vector[count_selected_index] << endl;
	if (count_vector[count_selected_index] >= 1)
		return rms_min_index[count_selected_index];
	else
		return rms_min_index[0];
}

int
 main (int argc, char** argv)
{
	//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª	
	if (argc < 3)
	{
		PCL_ERROR("Insufficient number of input variables! \n");
		printUsage(argv[0]);
		return -1;
	}
	//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª	

	//Default parameter area
	double sample_num = 10000;
	double overlap = 0.3;     //overlap rate
	int registration_type = 2; //0 means 'whole to whole', 1 means 'part to whole', other means 'part to part'
	int inputnum = 3;
	int part_nums = 0;
	double factor = 0.2;  //The larger the value, the greater the downsampling force, resulting in fewer points remaining after downsampling
//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª		
	//Update parameter area
	while (inputnum < argc) {
		if (!strcmp(argv[inputnum], "-o")) {
			overlap = atof(argv[++inputnum]);
			cout << "The overlapping area ratio is" << overlap << endl;
		}
		else if (!strcmp(argv[inputnum], "-s")) {
			sample_num = atoi(argv[++inputnum]);
			cout << "The number of downsampling is" << sample_num << endl;
		}
		else if (!strcmp(argv[inputnum], "-p")) {
			part_nums = atoi(argv[++inputnum]);
			cout << "Set the number of blocks to" << part_nums << endl;
		}
		else if (!strcmp(argv[inputnum], "-f")) {
			factor = atof(argv[++inputnum]);
			cout << "The impact factor during downsampling is" << factor << endl;
		}
		else if (!strcmp(argv[inputnum], "-r")) {
			registration_type = atoi(argv[++inputnum]);
			if (registration_type == 0) {
				cout << "registration type is 'whole to whole'" << endl;
			}
			else if (registration_type == 1) {
				cout << "registration type is 'part to whole'" << endl;
			}
			else {
				cout << "registration type is 'part to part'" << endl;
			}
		}
		else if (argv[inputnum][0] == '-') {
			std::cerr << "Unknown flag\n";
			printUsage(argv[0]);
			return -1;
		};
		inputnum++;
	}

//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª	
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr source_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // Source point cloud
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr target_point_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // Target point cloud
	//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª	
	//Read two input point clouds
	read_pointcloud(source_point_cloud, target_point_cloud, argc, argv);
	clock_t registration_start = clock();
	string pc1name = argv[1];
	string pc2name = argv[2];

	double source_resolution = 0.0;
	double target_resolution = 0.0;
	double high_resolution = 0.0;
	source_resolution = computeCloudResolution(source_point_cloud);
	target_resolution = computeCloudResolution(target_point_cloud);
	cout << "source_point_cloud point cloud resolution:" << source_resolution << endl;
	cout << "target_point_cloud point cloud resolution:" << target_resolution << endl;

	high_resolution = max(source_resolution, target_resolution);
	//Downsampling, the two point clouds become the same point cloud density
	pcl::VoxelGrid<pcl::PointXYZRGBA> src_voxel_grid;
	src_voxel_grid.setLeafSize(high_resolution * factor, high_resolution * factor, high_resolution * factor);
	//voxel_grid.setLeafSize(0.01, 0.01, 0.01);
	src_voxel_grid.setInputCloud(source_point_cloud);
	src_voxel_grid.filter(*source_point_cloud);

	pcl::VoxelGrid<pcl::PointXYZRGBA> tgt_voxel_grid;
	tgt_voxel_grid.setLeafSize(high_resolution * factor, high_resolution * factor, high_resolution * factor);
	//voxel_grid.setLeafSize(0.01, 0.01, 0.01);
	tgt_voxel_grid.setInputCloud(target_point_cloud);
	tgt_voxel_grid.filter(*target_point_cloud);

	source_resolution = computeCloudResolution(source_point_cloud);
	target_resolution = computeCloudResolution(target_point_cloud);
	cout << "Number of source_point_cloud point clouds after downsampling:" << source_point_cloud->points.size() << endl;
	cout << "Number of target_point_cloud point clouds after downsampling:" << target_point_cloud->points.size() << endl;
	cout << "Source_point_cloud point cloud resolution after downsampling:" << source_resolution << endl;
	cout << "Target_point_cloud point cloud resolution after downsampling:" << target_resolution << endl;
//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª	

	if (registration_type == 1) {
		overlap = double(target_point_cloud->points.size() )/ double(source_point_cloud->points.size());
		cout << "The calculated overlap area ratio is" << overlap << endl;
	}

	int src_seg_nums = floor(1/overlap*5);         // ÇÐ¸îµãÔÆµÄÊýÁ¿£¬Ä¬ÈÏÊÇ11
	cout << "According to the proportion of overlapping areas, the recommended number of blocks is" << src_seg_nums << endl;
	if(part_nums!=0)
	src_seg_nums = part_nums;
	double K_factor = overlap* src_seg_nums;     //·Ö¸îµÄÊ±ºòÐèÒªÉèÖÃµÄÓ°ÏìÒò×Ó£¬¾ö¶¨KdÊ÷µÄ´óÐ¡
	int tgt_seg_nums = 1;
	if ((registration_type != 0)&& (registration_type != 1)) {
		tgt_seg_nums = ceil(double(target_point_cloud->points.size()*src_seg_nums) / double(source_point_cloud->points.size()));
		src_seg_nums = ceil(double(source_point_cloud->points.size()*tgt_seg_nums)/ double(target_point_cloud->points.size()));
	}
	cout << "The final number of source_pointcloud blocks is" << src_seg_nums << endl;
	cout << "The final target_pointcloud block number is" << tgt_seg_nums << endl;

  //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
  //Display the original point cloud
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("Origin point cloud"));
  int v1(0);
  int v2(0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> source_cloud_color_handler(source_point_cloud, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> target_cloud_color_handler(target_point_cloud, 255, 0, 0);
  viewer1->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
  viewer1->setBackgroundColor(255, 255, 255, v1);
  viewer1->addText("source_point_cloud_image", 10, 30,1.0,0.0,0.0, "v1 text",v1);
  viewer1->addPointCloud(source_point_cloud, source_cloud_color_handler,"source_point_cloud", v1);
  viewer1->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
  viewer1->setBackgroundColor(255, 255, 255, v2);
  viewer1->addText("target_point_cloud_image", 10, 10, 1.0, 0.0, 0.0, "v2 text", v2);
  viewer1->addPointCloud(target_point_cloud, target_cloud_color_handler, "target_point_cloud", v2);

//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
  //display point cloud blocks
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("source point cloud block perspective"));
  viewer2->setBackgroundColor(255, 255, 255); 
  viewer2->addText("seg_source_point_cloud_image,total " + to_string(src_seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0);
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer3(new pcl::visualization::PCLVisualizer("Target point cloud block perspective"));
  viewer3->setBackgroundColor(255, 255, 255); 
  viewer3->addText("seg_target_point_cloud_image,total " + to_string(tgt_seg_nums) + "parts", 10, 10, 1.0, 0.0, 0.0);
 
  vector<int> source_sampleIndex;
  vector<int> target_sampleIndex;
  ComputeFPS(source_point_cloud, src_seg_nums, source_sampleIndex);
  ComputeFPS(target_point_cloud, tgt_seg_nums, target_sampleIndex);
 
  pcl::PointCloud<pcl::PointXYZRGBA> PointCloud;
  vector<pcl::PointCloud<pcl::PointXYZRGBA>> NewSourceCloud(src_seg_nums, PointCloud);
  pcl::PointCloud<pcl::PointXYZRGBA> PointCloud1;
  vector<pcl::PointCloud<pcl::PointXYZRGBA>> NewTargetCloud(tgt_seg_nums, PointCloud1);

  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PointCloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
  vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> TransformCloud(src_seg_nums*tgt_seg_nums, PointCloud2);
  
  Eigen::Matrix4f ransactransformation;
  Eigen::Matrix4f icptransformation;
  Eigen::Matrix4f finaltransformation;

  vector<Eigen::Matrix4f> TR_vector(src_seg_nums*tgt_seg_nums);

  double rms;
  double finalrms = 0.0;

  // Put the Similarity value into a vector array
  vector<double> rms_vector(src_seg_nums*tgt_seg_nums, -1);

  //Save the point cloud sequence number of the selected area
  vector<vector<int>> NewSourceCloudIndex(src_seg_nums);
  vector<vector<int>> NewTargetCloudIndex(tgt_seg_nums);
  vector<vector<int>> PointRegionPair(src_seg_nums*tgt_seg_nums);
  int K = (int)(min(source_point_cloud->points.size() / src_seg_nums,target_point_cloud->points.size()/ tgt_seg_nums)* K_factor); //Number of point clouds contained in each block
  //cout << "Number of point clouds contained in each block:"<<K << endl; 

  string colorname[16] = { "red","orange","yellow","green","cyan","blue","purple","darkRed","darkOrange","gold","oliveDrab","DarkTurquoise","DarkSlateBlue","DarkViolet","Pink", "Brown4"};
  double r[16] = {255,255,255,0,0,0,160,139,255,255,107,0,72,148,255,139};
  double g[16] = {0,165,255,255,255,0,32,0,127,215,142,206,61,0,192,35};
  double b[16] = {0,0,0,0,255,255,240,0,0,0,35,209,139,211,203,35};
  clock_t segment_start = clock();
#pragma omp parallel for
  for (int i = 0; i < src_seg_nums; i++)
  {
	  SegmentCloud(source_point_cloud, NewSourceCloud[i], NewSourceCloudIndex[i], source_sampleIndex[i], src_seg_nums, K, sample_num);
  }
  for (int i = 0; i < tgt_seg_nums; i++)
  {
	  SegmentCloud(target_point_cloud, NewTargetCloud[i], NewTargetCloudIndex[i], target_sampleIndex[i], tgt_seg_nums, K, sample_num);
  }
  clock_t segment_end = clock();
  cout << "Finish point cloud block, time:" << (double)(segment_end - segment_start) / (double)CLOCKS_PER_SEC << " s" << endl;
  for (int i = 0; i < src_seg_nums; i++)
  {
	  boost::shared_ptr<vector<int>> index_partptr1 = boost::make_shared<vector<int>>(NewSourceCloudIndex[i]);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc1PartRegion(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::ExtractIndices<pcl::PointXYZRGBA> part1Indice;
	  part1Indice.setInputCloud(source_point_cloud);
	  part1Indice.setIndices(index_partptr1);
	  part1Indice.setNegative(false);                    
	  part1Indice.filter(*pc1PartRegion);
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> part_source_pointcloud_color_handler(pc1PartRegion, r[i], g[i], b[i]);//Different blocks give different colors
	  viewer2->addPointCloud(pc1PartRegion, part_source_pointcloud_color_handler, "source_point_cloud" + to_string(i));
	  viewer2->addText("part" + to_string(i), 10 + i * 50, 30, r[i] / 255, g[i] / 255, b[i] / 255);
  }
  cout << "Finish "<<src_seg_nums<<" parts display of source point cloud"  << endl;
  for (int i = 0; i < tgt_seg_nums; i++)
  {

	  boost::shared_ptr<vector<int>> index_partptr2 = boost::make_shared<vector<int>>(NewTargetCloudIndex[i]);
	  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc2PartRegion(new pcl::PointCloud<pcl::PointXYZRGBA>);
	  pcl::ExtractIndices<pcl::PointXYZRGBA> part2Indice;
	  part2Indice.setInputCloud(target_point_cloud);
	  part2Indice.setIndices(index_partptr2);
	  part2Indice.setNegative(false);                    
	  part2Indice.filter(*pc2PartRegion);
	  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> part_target_pointcloud_color_handler(pc2PartRegion, r[i], g[i], b[i]);//Different blocks give different colors
	  viewer3->addPointCloud(pc2PartRegion, part_target_pointcloud_color_handler, "target_point_cloud" + to_string(i));
	  viewer3->addText("part" + to_string(i), 10 + i * 50, 30, r[i] / 255, g[i] / 255, b[i] / 255);
  }
  cout << "Finish " << tgt_seg_nums <<" parts display of target point cloud"  << endl;

  //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
   clock_t ransac_start = clock();
  //µãÔÆ·Ö¿éºóransacÅä×¼
	//¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª

#pragma omp parallel for
	   for (int i = 0; i < src_seg_nums; i++) {
		   for (int j = 0; j < tgt_seg_nums; j++) {
			   Ransac_registration(NewSourceCloud[i].makeShared(), NewTargetCloud[j].makeShared(), TR_vector[i*tgt_seg_nums + j], rms_vector[i*tgt_seg_nums + j]);
			   PointRegionPair[i*tgt_seg_nums + j] = { i,j };
			   cout << "The similarity between source cloud part" << i << "and target point cloud part " << j << ",the similarity is "<<rms_vector[i*tgt_seg_nums + j]<<"¡£The thread used is:"<< omp_get_thread_num() << endl;

		   }
	   }
  clock_t ransac_end = clock();
  cout << "Similarity calculation ends, time used:" << (double)(ransac_end - ransac_start) / (double)CLOCKS_PER_SEC << " s" << endl;

  //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
  //Use the TAM module to find the appropriate block pair
  int res_min_pair = ComputerRelativeTR_Threshold(rms_vector, TR_vector, high_resolution,PointRegionPair);
  double rms_min = rms_vector[res_min_pair];
  ransactransformation = TR_vector[res_min_pair];
  //Extract block pair
  int pc1;
  int pc2;
  vector<int> pcpair;
  pcpair = PointRegionPair[res_min_pair];
  pc1 = pcpair[0];
  pc2 = pcpair[1];


  cout << "The similarity of the output matrix£º" << rms_min << endl;
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pair1ransacCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pair1icpCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr OutCloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  cout << "pc1£º" << pc1 << endl;
  cout << "pc2£º" << pc2 << endl;

  boost::shared_ptr<vector<int>> index_ptr1_b = boost::make_shared<vector<int>>(NewSourceCloudIndex[pc1]);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc1PairRegion_b(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::ExtractIndices<pcl::PointXYZRGBA> pair1Indice_b;
  pair1Indice_b.setInputCloud(source_point_cloud);
  pair1Indice_b.setIndices(index_ptr1_b);
  pair1Indice_b.setNegative(false);                    
  pair1Indice_b.filter(*pc1PairRegion_b);


  boost::shared_ptr<vector<int>> index_ptr2_b = boost::make_shared<vector<int>>(NewTargetCloudIndex[pc2]);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc2PairRegion_b(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::ExtractIndices<pcl::PointXYZRGBA> pair2Indice_b;
  pair2Indice_b.setInputCloud(target_point_cloud);
  pair2Indice_b.setIndices(index_ptr2_b);
  pair2Indice_b.setNegative(false);                    
  pair2Indice_b.filter(*pc2PairRegion_b);
  //Register according to the point block pair obtained by TAM
  method_sac_icp(pc1PairRegion_b, pc2PairRegion_b, finaltransformation);
  pcl::transformPointCloud(*source_point_cloud, *OutCloud, finaltransformation);
  clock_t registration_end = clock();
  //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
  //Compute the error
  Eigen::Matrix4f trueMatrix; //The truth matrix of the transformation set here can be used to evaluate the accuracy
  trueMatrix << 0.813808, -0.34201, 0.469835, 0,
	  0.296192, 0.939696, 0.171001, 0,
	  -0.499987, 0, 0.866033, 0,
	  0, 0, 0, 1;
  double Rerror = 0.0;
  double Terror = 0.0;
  compute_error(finaltransformation, trueMatrix, Rerror, Terror);
  cout << "The rotation error is: " << Rerror << endl;
  cout << "The translation error is: " << Terror << endl;
  //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
  cout << "registration time: " << (double)(registration_end - registration_start) / (double)CLOCKS_PER_SEC << " s" << endl;
  //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª
  //TRMS×îÐ¡µÄÇøÓò	
  boost::shared_ptr<vector<int>> index_ptr1 = boost::make_shared<vector<int>>(NewSourceCloudIndex[pc1]);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc1PairRegion(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::ExtractIndices<pcl::PointXYZRGBA> pair1Indice;
  pair1Indice.setInputCloud(OutCloud);
  pair1Indice.setIndices(index_ptr1);
  pair1Indice.setNegative(false);                    
  pair1Indice.filter(*pc1PairRegion);


  boost::shared_ptr<vector<int>> index_ptr2 = boost::make_shared<vector<int>>(NewTargetCloudIndex[pc2]);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pc2PairRegion(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::ExtractIndices<pcl::PointXYZRGBA> pair2Indice;
  pair2Indice.setInputCloud(target_point_cloud);
  pair2Indice.setIndices(index_ptr2);
  pair2Indice.setNegative(false);                    
  pair2Indice.filter(*pc2PairRegion);

  //-------------------------------------------------------------------------------------------------------------------------------------
 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer4(new pcl::visualization::PCLVisualizer("Ñ¡È¡µÄµãÔÆÇøÓò"));
 viewer4->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
 viewer4->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
 viewer4->setBackgroundColor(255, 255, 255, v1);
 viewer4->setBackgroundColor(255, 255, 255, v2);
 viewer4->addPointCloud(source_point_cloud, "source_point_cloud", v1);
 viewer4->addPointCloud(target_point_cloud, "target_point_cloud", v2);
 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> sourcePairRegion_cloud_color_handler(pc1PairRegion_b, r[pc1], g[pc1], b[pc1]);
 viewer4->addPointCloud(pc1PairRegion_b ,sourcePairRegion_cloud_color_handler, "sourcePairRegion", v1);
 viewer4->addText("source_point_cloud_image,total " + to_string(src_seg_nums) + "parts", 10, 30, 1.0, 0.0, 0.0, "v1 text", v1);
 viewer4->addText("select part" + to_string(pc1), 10 , 10, r[pc1] / 255, g[pc1] / 255, b[pc1] / 255, "v1 select text", v1);
 pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> targetPairRegion_cloud_color_handler(pc2PairRegion_b, r[pc2], g[pc2], b[pc2]);
 viewer4->addPointCloud(pc2PairRegion_b, targetPairRegion_cloud_color_handler, "targetPairRegion", v2);
 viewer4->addText("target_point_cloud_image,total " + to_string(tgt_seg_nums) + "parts", 10, 30, 1.0, 0.0, 0.0, "v2 text", v2);
 viewer4->addText("select part" + to_string(pc2), 10 , 10, r[pc2] / 255, g[pc2] / 255, b[pc2] / 255, "v2 select text", v2);
 viewer4->spinOnce(100);

  // Show point cloud test
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Visualize the source and target point clouds after registration by our algorithm"));
  viewer->setBackgroundColor(255, 255, 255);
  viewer->addText("source cloud select part" + to_string(pc1), 10, 10, r[pc1] / 255, g[pc1] / 255, b[pc1] / 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> best_sourcePairRegion_cloud_color_handler(pc1PairRegion, r[pc1], g[pc1], b[pc1]);
  viewer->addPointCloud(OutCloud,  "source_point_cloud");
  viewer->addPointCloud(pc1PairRegion, best_sourcePairRegion_cloud_color_handler, "sourcePairRegion");
  viewer->addText("target cloud select part" + to_string(pc2), 10, 30, r[pc2] / 255, g[pc2] / 255, b[pc2] / 255);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGBA> best_targetPairRegion_cloud_color_handler(pc2PairRegion, r[pc2], g[pc2], b[pc2]);
  viewer->addPointCloud(target_point_cloud, "target_point_cloud");
  viewer->addPointCloud(pc2PairRegion, best_targetPairRegion_cloud_color_handler, "targetPairRegion_cloud");

 //¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª¡ª

  //Save transform matrix
  //cout << pc1name << endl;
  int begin = pc2name.find_last_of('\\');
  string origin_transform_matrix_filepath = pc1name.substr(0, pc1name.size() - 4) + "_part" + to_string(pc1) + "_" + pc2name.substr(begin+1, pc2name.size() - 4) + "_part" + to_string(pc2) + ".txt";
  char ExePath[MAX_PATH];
  GetModuleFileName(NULL, ExePath, MAX_PATH);
  cout << "Transformation matrix save path£º" << ExePath << origin_transform_matrix_filepath << endl;
  std::ofstream  origin;
  origin.open(origin_transform_matrix_filepath, std::ios::app);//Append write at the end of the file
  cout << finaltransformation << endl;
  origin << finaltransformation << std::endl;//New line after each matrix is written
  origin.close();
  //savePointCloud
  savePointCloud(OutCloud,"output.ply");
  while (!viewer->wasStopped())
  {
	  viewer->spinOnce(100);
	  viewer1->spinOnce(100);
	  viewer2->spinOnce(100);
	  viewer3->spinOnce(100);
	  viewer4->spinOnce(100);
  }
 return (0);
}
