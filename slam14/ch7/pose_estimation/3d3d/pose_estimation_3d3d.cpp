#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

void find_feature_matches
(	const Mat &img1,const Mat &img2,
	vector<KeyPoint> &keypoints_1,
	vector<KeyPoint> &keypoints_2,
	vector<DMatch> &matches
);

void pose_estimation_3d3d
(	const vector<Point3f> &pts1,
	const vector<Point3f> &pts2,
	Mat &R,Mat &t
	);

Point2d pixel_cam(const Point2d &p,const Mat &K);

int main(int argc,char **argv)
{
	if(argc!=5)
	{
		cout<<"usage:pose_estimation_3D2D img1 img2 depth1 depth2"<<endl;
		return 1;
	}
	Mat img1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
	Mat img2=imread(argv[2],CV_LOAD_IMAGE_COLOR);
	vector<KeyPoint> keypoints_1,keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img1,img2,keypoints_1,keypoints_2,matches);

	Mat K=(Mat_<double>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
	Mat depth1=imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
	Mat depth2=imread(argv[4],CV_LOAD_IMAGE_UNCHANGED);
	vector<Point3f> pts1,pts2;
    for ( DMatch m:matches )
    {
        ushort d1 = depth1.ptr<unsigned short> ( int ( keypoints_1[m.queryIdx].pt.y ) ) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort d2 = depth2.ptr<unsigned short> ( int ( keypoints_2[m.trainIdx].pt.y ) ) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( d1==0 || d2==0 )   // bad depth
            continue;
        Point2d p1 = pixel_cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel_cam ( keypoints_2[m.trainIdx].pt, K );
        float dd1 = float ( d1 ) /5000.0;
        float dd2 = float ( d2 ) /5000.0;
        pts1.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );
        pts2.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }

	Mat R,t;
	pose_estimation_3d3d(pts1,pts2,R,t);
	cout<<"R="<<R<<endl;
	cout<<"t="<<t<<endl;
	cout<<"R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;

	return 0;

}
void pose_estimation_3d3d
(	const vector<Point3f> &pts1,
	const vector<Point3f> &pts2,
	Mat &R,Mat &t
	)
{
	Point3f p1,p2;
	int N=pts1.size();
	for(int i=0;i<N;++i)
	{
		p1+=pts1[i];
		p2+=pts2[i];
	}
	p1/=N;p2/=N;
	vector<Point3f> q1(N),q2(N);
	for(int i=0;i<N;++i)
	{
		q1[i]=pts1[i]-p1;
		q2[i]=pts2[i]-p2;
	}

	Matrix3d W=Matrix3d::Zero();
	for(int i=0;i<N;++i)
	{
		W+=Vector3d(q1[i].x,q1[i].y,q1[i].z)
		*Vector3d(q2[i].x,q2[i].y,q2[i].z).transpose();
	}
	cout<<"W="<<W<<endl;
	JacobiSVD<Matrix3d>svd(W,ComputeFullU|ComputeFullV);
	Matrix3d U=svd.matrixU();
	Matrix3d V=svd.matrixV();
	cout<<"U="<<U<<endl;
	cout<<"V="<<V<<endl;
	Matrix3d R_=U*(V.transpose());
	Vector3d t_=Vector3d(p1.x,p1.y,p1.z)-R_*Vector3d(p2.x,p2.y,p2.z);

	R=(Mat_<double>(3,3)<<
		R_(0,0),R_(0,1),R_(0,2),
		R_(1,0),R_(1,1),R_(1,2),
		R_(2,1),R_(2,1),R_(2,2)
		);
	t=(Mat_<double>(3,1)<<t_(0,0),t_(1,0),t_(2,0));
}	

void find_feature_matches
(	const Mat &img1,const Mat &img2,
	vector<KeyPoint> &keypoints_1,
	vector<KeyPoint> &keypoints_2,
	vector<DMatch> &matches
)
{
	Mat descriptors_1,descriptors_2;
	vector<DMatch> originMatch;
	Ptr<ORB> orb=ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);

	orb->detect(img1,keypoints_1);
	orb->detect(img2,keypoints_2);

	orb->compute(img1,keypoints_1,descriptors_1);
	orb->compute(img2,keypoints_2,descriptors_2);

	BFMatcher matcher (NORM_HAMMING);
	matcher.match(descriptors_1,descriptors_2,originMatch);

	double min_dist=10000,max_dist=0;
	for(int i=0;i<descriptors_1.rows;i++)
	{
		double dist=originMatch[i].distance;
		if(dist<min_dist) min_dist=dist;
		if(dist>max_dist) max_dist=dist;
	}

	printf("--Max dist:%f \n",max_dist);
	printf("--Min dist:%f \n",min_dist);

	for(int i=0;i<descriptors_1.rows;i++)
		if(originMatch[i].distance<=max(2*min_dist,30.0))
		{
			matches.push_back(originMatch[i]);
		}
}

Point2d pixel_cam(const Point2d &p,const Mat &K)
{
	return Point2d
	(
		(p.x-K.at<double>(0,2))/K.at<double>(0,0),
		(p.y-K.at<double>(1,2))/K.at<double>(1,1)
	);

}