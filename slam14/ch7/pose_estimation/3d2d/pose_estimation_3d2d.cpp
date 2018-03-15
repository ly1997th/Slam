#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches
(	const Mat &img1,const Mat &img2,
	vector<KeyPoint> &keypoints_1,
	vector<KeyPoint> &keypoints_2,
	vector<DMatch> &matches
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

	Mat d1=imread(argv[3],CV_LOAD_IMAGE_UNCHANGED);
	Mat K=(Mat_<double>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
	vector<Point3f> pts_3d;
	vector<Point2f> pts_2d;

	for(DMatch m:matches)
	{
		ushort d=d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
		if(d==0) continue;
		float dd=d/1000.0;
		Point2d p1=pixel_cam(keypoints_1[m.queryIdx].pt,K);
		pts_3d.push_back(Point3f(p1.x*dd,p1.y*dd,dd));
		pts_2d.push_back(keypoints_2[m.trainIdx].pt);
	}
	cout<<"3d-2d pairs:"<<pts_3d.size()<<endl;
	Mat r,t;
	solvePnP(pts_3d,pts_2d,K,Mat(),r,t,false,cv::SOLVEPNP_EPNP);
	Mat R;
	cv::Rodrigues(r,R);

	cout<<"R="<<endl<<R<<endl;
	cout<<"t="<<endl<<t<<endl;
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