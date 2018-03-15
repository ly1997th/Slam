#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;


void find_feature_matches
(	const Mat &img1,const Mat &img2,
	vector<KeyPoint> &keypoints_1,
	vector<KeyPoint> &keypoints_2,
	vector<DMatch> &matches
);

void pose_estimation_2d2d
(
	std::vector<KeyPoint> keypoints_1,
	std::vector<KeyPoint> keypoints_2,
	std::vector<DMatch> matches,
	Mat& R,Mat& t
);

Point2d pixel_cam(const Point2d &p,const Mat &K);

int main(int argc,char** argv)
{
	if(argc!=3)
	{
		cout<<"usage:feature_extraction img1 img2"<<endl;
		return 1;
	}

	Mat img_1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
	Mat img_2=imread(argv[2],CV_LOAD_IMAGE_COLOR);

	vector<KeyPoint> keypoints_1,keypoints_2;
	vector<DMatch> matches;
	find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
	cout<<"Totally,"<<matches.size()<<"pairs of matched points are found"<<endl;

	Mat R,t;
	pose_estimation_2d2d(keypoints_1,keypoints_2,matches,R,t);

	//tansform t vector into li matrix
	Mat t_x=(Mat_<double>(3,3)<<        
		0,  -t.at<double>(2,0), t.at<double>(1,0),
		t.at<double>(2,0),  0, -t.at<double>(0,0),
		-t.at<double>(1,0), t.at<double>(0,0), 0);
	cout<<"t^R="<<endl<<t_x*R<<endl;

	Mat K=(Mat_<double>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
	for(DMatch m:matches)
	{
		Point2d pt1=pixel_cam(keypoints_1[m.queryIdx].pt,K);
		Mat y1=(Mat_<double>(3,1)<<pt1.x,pt1.y,1);
		Point2d pt2=pixel_cam(keypoints_2[m.trainIdx].pt,K);
		Mat y2=(Mat_<double>(3,1)<<pt2.x,pt2.y,1);
		Mat d=y2.t()*t_x*R*y1;
		cout<<"epipolar constraint="<<d<<endl;
	}

	return 0;
}


void pose_estimation_2d2d
(
	std::vector<KeyPoint> keypoints_1,
	std::vector<KeyPoint> keypoints_2,
	std::vector<DMatch> matches,
	Mat& R,Mat& t
)
{
	Mat K=(Mat_<double> (3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
	vector<Point2f> points1;
	vector<Point2f> points2;

	for(int i=0;i<(int) matches.size();i++)
	{
		points1.push_back(keypoints_1[matches[i].queryIdx].pt);
		points2.push_back(keypoints_2[matches[i].trainIdx].pt);
	}

	Mat fundamental_matrix;
	//use eight points to calculate fundamental matrix
	fundamental_matrix=findFundamentalMat(points1,points2,CV_FM_8POINT);
	cout<<"fundamental_matrix is"<<endl<<fundamental_matrix<<endl;
	Point2d principal_point(325.1,249.7);
	int focal_length=521;
	Mat essential_matrix;
	essential_matrix=findEssentialMat(points1,points2,focal_length,principal_point,RANSAC);
	cout<<"essential_matrix is"<<endl<<essential_matrix<<endl;

	Mat homography_matrix;
	homography_matrix=findHomography(points1,points2,RANSAC,3,noArray(),2000,0.99);
	cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

	recoverPose(essential_matrix,points1,points2,R,t,focal_length,principal_point);
	cout<<"R is"<<endl<<R<<endl;
	cout<<"t is"<<endl<<t<<endl;
}

void find_feature_matches(const Mat &img1,const Mat &img2,
						vector<KeyPoint> &keypoints_1,
						vector<KeyPoint> &keypoints_2,
						vector<DMatch> &matches)
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
		if(dist<max_dist) max_dist=dist;
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