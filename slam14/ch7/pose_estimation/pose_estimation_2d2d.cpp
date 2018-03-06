#include <iostream>





using namespace std;
using namespace cv;

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
	find_features_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
	cout<<"Totally,"<<matches.size()<<"pairs of matched points are found"<<endl;

	Mat R,t;
	pose_estimation_2d2d(keypoints_1,keypoints_2,matches,R,t);

	//tansform t vector into li matrix
	Mat t_x=(Mat_<double>(3,3)<<        
		0,  -t.at<double>(2,0), t.at<double>(1,0),
		t.at<double>(2,0),  0, -t.at<double>(0,0),
		-t.at<double>(1,0) t.at<double>(0,0) 0);
	cout<<"t^R="<<endl<<t_x*R<<endl;

	Mat K=(Mat_<doubel>(3,3)<<520.9,0,325.1,0,521.0,249.7,0,0,1);
	for(DMatch m:matches)
	{
		Point2d pt1=pixel2cam(keypoints_1[m.queryIdx].pt,K);
		Mat y1=(Mat_<double>(3,1)<<pt1.x,pt1.y,1);
		Point2d pt2=pixel2cam(keypoints_2[m.trainIdx].pt,K);
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
		points2.push_back(keypoints_2[matches[i].trainIdx.pt]);
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