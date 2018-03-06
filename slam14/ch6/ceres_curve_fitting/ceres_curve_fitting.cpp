#include<iostream> 
#include<opencv2/core/core.hpp>
#include<ceres/ceres.h>
#include<chrono>

using namespace std;

struct CURVE_FITTING_COST
{
	CURVE_FITTING_COST(double x,double y):_x(x),_y(y){}//calculate residual
	template<typename T>
	bool operator() (const T* const abc,T* residual) const
	{
		//y=exp(ax^2+bx+c)
		residual[0]=T(_y)-ceres::exp(abc[0]*T(_x)*T(_x)+abc[1]*T(_x)+abc[2]);
		return true;
	} 
	const double _x,_y;

};

int main(int argc,char **argv)
{
	double a=1.0,b=2.0,c=1.0;// real parameter value
	int N=100;				//data point
	double w_sigma=1.0; 		//noise value
	cv::RNG rng;			//OpenCV random number 
	double abc[3]={0,0,0};	//abc parameter prediction

	vector<double> x_data,y_data;

	cout<<"generating data:"<<endl;
	for(int i=0;i<N;++i)
	{
		double x=i/100.0;
		x_data.push_back(x);
		y_data.push_back(exp(a*x*x+b*x+c)+rng.gaussian(w_sigma));
		cout<<x_data[i]<<" "<<y_data[i]<<endl;
	}
	//构建最小二乘问题
	ceres::Problem problem;
	for(int i=0;i<N;++i)
	{
		problem.AddResidualBlock		//向问题中添加误差项
		(
			//使用自动求导，模板参数：误差类型，输出维度，输入维度，数值参考前面struct中写法
			new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>
			(
				new CURVE_FITTING_COST (x_data[i],y_data[i])
			),
			nullptr,						//核函数，这里不使用，为空
			abc    							//带估计参数
		);
	}
	//配置求解器
	ceres::Solver::Options options;					//中文测试
	options.linear_solver_type=ceres::DENSE_QR;		//增量方程如何求解
	options.minimizer_progress_to_stdout=true;		//输出到cout

	ceres::Solver::Summary summary;					//优化信息
	chrono::steady_clock::time_point t1=chrono::steady_clock::now();
	ceres::Solve(options,&problem,&summary);		//开始优化
	chrono::steady_clock::time_point t2=chrono::steady_clock::now();
	chrono::duration<double>time_used
	=chrono::duration_cast<chrono::duration<double>>(t2-t1);

	cout<<"solve time cost="<<time_used.count()<<"seconds."<<endl;

	cout<<summary.BriefReport()<<endl;
	cout<<"estimated a,b,c=";
	for(auto p:abc) cout<<p<<" ";
	cout<<endl;

	return 0;
}