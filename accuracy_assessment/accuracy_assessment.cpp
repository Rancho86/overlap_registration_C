#include<conio.h>
#include<vector>
#include<iostream>
#include<string>
#include<ctime>
#include<algorithm>
#include<math.h>
#include<omp.h> //并行计算
#include <fstream>

#include <pcl/io/io.h>
#include<pcl/console/parse.h>

//

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>	

using namespace std;


/*
void printUsage(const char* progName)
{
	cout << "先输入点云名字，再加一个浮点数字0是手动，1是sac_icp,不输入默认为0" << endl;
	cout << "例子：" << progName << " source_point_cloud.pcd/obj/ply target_point_cloud.pcd/obj/ply 1" << endl;
	cout << "测试版本" << endl;
}
*/
void ReadData(std::istream &fin, Eigen::Matrix4f &m_matrix)
{
	int numRow = m_matrix.rows();
	int numCol = m_matrix.cols();

	for (int j = 0; j < numRow; j++)//共numRow行
	{
		for (int i = 0; i < numCol; i++)//共numCol列组成一行
		{
			fin >> m_matrix(j, i);
		}

	}
}

void compute_error(const Eigen::Matrix4f TR_registration, const Eigen::Matrix4f TR_true, double &Rerror, double &Terror) {
	Eigen::Matrix4f TR_error;
	TR_error = TR_registration * TR_true.inverse();
	cout << TR_error << endl;
	Eigen::Matrix3f TR_Diff_R;
	TR_Diff_R= TR_error.block<3, 3>(0, 0);
	cout << TR_Diff_R << endl;
	Rerror = fabs(acos((TR_Diff_R.trace() - 1) / 2)) * 180 / 3.14;
	Terror = (fabs(TR_error(0, 3)) + fabs(TR_error(1, 3)) + fabs(TR_error(2, 3))) / 3;
}



int
main(int argc, char** argv)
{
	Eigen::Matrix4f TR_registration;
	Eigen::Matrix4f TR_true;
	double Rerror = 0;
	double Terror = 0;
	if (argc > 2) {


	//  Eigen::Matrix4f icptransformation;
	std::ifstream registration_matrix(argv[1], std::ios::binary);
	if (!registration_matrix)
	{
		return 0;
	}
	ReadData(registration_matrix, TR_registration);
	std::ifstream true_matrix(argv[2], std::ios::binary);
	if (!true_matrix)
	{
		return 0;
	}
	ReadData(true_matrix, TR_true);
	cout << "matrix1为：" << TR_registration << endl;
	cout << "matrix2为：" << TR_true << endl;
	compute_error(TR_registration, TR_true, Rerror, Terror);
	cout << "旋转误差为：" << Rerror << endl;
	cout << "平移误差为：" << Terror << endl;
	}	
	return (0);
}