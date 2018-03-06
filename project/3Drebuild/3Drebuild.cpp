#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/legacy/legacy.hpp>
using namespace cv;
using namespace std;

void extract_features(
	//vector<string>& img1,
	//vector<string>& img2,
	Mat &image1,
	Mat &image2,
	vector<KeyPoint>& keypoints1,
	vector<KeyPoint>& keypoints2,
	Mat & descriptor1,
	Mat & descriptor2,
	vector<Vec3b >& colors1,
	vector<Vec3b >& colors2
	)
{
	keypoints1.clear();keypoints2.clear();
	//parameter test 
	// imshow("image1",image1);
	// imshow("image2",image2);
	// waitKey(0);

	Ptr<ORB> orb=ORB::create(500,1.2f,8,31,0,2,ORB::HARRIS_SCORE,31,20);
	//orb using example
	// orb->detect(img_1,keypoints_1);
	// orb->detect(img_2,keypoints_2);

	// orb->compute(img_1,keypoints_1,descriptors_1);
	// orb->compute(img_2,keypoints_2,descriptors_2);

	    //-- 第一步:检测 Oriented FAST 角点位置
	orb->detect ( image1,keypoints1 );
	orb->detect ( image2,keypoints2 );

	    //-- 第二步:根据角点位置计算 BRIEF 描述子
	orb->compute ( image1, keypoints1, descriptor1 );
	orb->compute ( image2, keypoints2, descriptor2 );
	
	//show the feature points 
	 // Mat outimg1;
  	//    drawKeypoints( image1, keypoints1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
 	 //    imshow("ORB特征点",outimg1);
  	//    waitKey(0);

  
	vector<Vec3b> colorsa(keypoints1.size());
	for (int i = 0; i < keypoints1.size(); ++i)
	{
		Point2f& p = keypoints1[i].pt;
		colorsa[i] = image1.at<Vec3b>(p.y, p.x);
	}
	colors1=colorsa;

	vector<Vec3b> colorsb(keypoints2.size());
	for (int i = 0; i < keypoints2.size(); ++i)
	{
		Point2f& p = keypoints2[i].pt;
		colorsb[i] = image2.at<Vec3b>(p.y, p.x);
	}
	colors2=colorsb;

}

void match_features(Mat& query, Mat& train, vector<DMatch>& matches)
{
	vector<vector<DMatch> > knn_matches;
	BFMatcher matcher(NORM_L2);
	matcher.knnMatch(query, train, knn_matches, 2);

	//获取满足Ratio Test的最小匹配的距离
	float min_dist = FLT_MAX;
	for (int r = 0; r < knn_matches.size(); ++r)
	{
		//Ratio Test
		//cout<<knn_matches[r][0].distance<<" ";
		if (knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance)
			continue;

		float dist = knn_matches[r][0].distance;
		if (dist < min_dist) min_dist = dist;
	}
	matches.clear();
	//cout<<min_dist;
	for (size_t r = 0; r < knn_matches.size(); ++r)
	{
		//排除不满足Ratio Test的点和匹配距离过大的点
		if (
			//knn_matches[r][0].distance > 0.6*knn_matches[r][1].distance ||
			knn_matches[r][0].distance > 2.5 * max(min_dist, .0f)
			)
			continue;

		//保存匹配点
		matches.push_back(knn_matches[r][0]);
	}
	
	cout<<"matched points number:"<<knn_matches.size()<<endl;
	cout<<"good matched points number:"<<matches.size()<<endl;
}

bool find_transform(Mat& K, vector<Point2f>& p1, vector<Point2f>& p2, Mat& R, Mat& T, Mat& mask)
{
	//根据内参矩阵获取相机的焦距和光心坐标（主点坐标）
	double focal_length = 0.5*(K.at<double>(0) + K.at<double>(4));
	Point2d principle_point(K.at<double>(2), K.at<double>(5));

	//根据匹配点求取本征矩阵，使用RANSAC，进一步排除失配点
	Mat E = findEssentialMat(p1, p2, focal_length, principle_point, RANSAC, 0.999,50, mask);

	//cout<<"test point 1"<<endl;
	if (E.empty()) return false;
	//cout<<"essential matrxi:"<<endl<<format(E,"csv")<<end<<endl;
	// randu(E,Scalar::all(0),Scalar::all(255));
	// cout<<"E (default)="<<endl<<E<<endl<<endl;

	double feasible_count = countNonZero(mask);
	cout << "RANSAC:"<<(int)feasible_count << " -in- " << p1.size() << endl;
	//对于RANSAC而言，outlier数量大于50%时，结果是不可靠的
	// if (feasible_count <= 15 || (feasible_count / p1.size()) < 0.6)
	// 	return false;
	//cout<<"test point 4"<<endl;

	//分解本征矩阵，获取相对变换
	//randu(R,Scalar::all(-500),Scalar::all(500));
	//cout<<"R(default)="<<endl<<R<<endl<<endl;
	int pass_count = recoverPose(E, p1, p2, R, T, focal_length, principle_point, mask);
	//cout<<pass_count;
	//randu(R,Scalar::all(-500),Scalar::all(500));
	//cout<<"R(default)="<<endl<<R<<endl<<endl;
	//同时位于两个相机前方的点的数量要足够大
	if (((double)pass_count) / feasible_count < 0.7)
		return false;

	return true;
}

void get_matched_points(
	vector<KeyPoint>& p1,
	vector<KeyPoint>& p2,
	vector<DMatch> matches,
	vector<Point2f>& out_p1,
	vector<Point2f>& out_p2
	)
{
	out_p1.clear();
	out_p2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_p1.push_back(p1[matches[i].queryIdx].pt);
		out_p2.push_back(p2[matches[i].trainIdx].pt);
	}
}

void get_matched_colors(
	vector<Vec3b>& c1,
	vector<Vec3b>& c2,
	vector<DMatch> matches,
	vector<Vec3b>& out_c1,
	vector<Vec3b>& out_c2
	)
{
	out_c1.clear();
	out_c2.clear();
	for (int i = 0; i < matches.size(); ++i)
	{
		out_c1.push_back(c1[matches[i].queryIdx]);
		out_c2.push_back(c2[matches[i].trainIdx]);
	}
}

void reconstruct(Mat& K, Mat& R, Mat& T, vector<Point2f>& p1, vector<Point2f>& p2, Mat& structure)
{
	//两个相机的投影矩阵[R T]，triangulatePoints只支持float型
	Mat proj1(3, 4, CV_32FC1);
	Mat proj2(3, 4, CV_32FC1);

	proj1(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_32FC1);
	proj1.col(3) = Mat::zeros(3, 1, CV_32FC1);
	//cout<<"test point 2"<<endl;
	//cout<<CV_32FC1;
	// randu(R,Scalar::all(-500),Scalar::all(500));
	// cout<<"R(default)="<<endl<<R<<endl<<endl;
	// randu(proj2,Scalar::all(0),Scalar::all(255));
	// cout<<"proj2 (default)="<<endl<<proj2<<endl<<endl;
	R.convertTo(proj2(Range(0, 3), Range(0, 3)), CV_32FC1);
	//cout<<"test point 3"<<endl;
	 T.convertTo(proj2.col(3), CV_32FC1);

	 Mat fK;
	 K.convertTo(fK, CV_32FC1);
	 proj1 = fK*proj1;
	 proj2 = fK*proj2;

	//三角重建
	triangulatePoints(proj1, proj2, p1, p2, structure);
}

void maskout_points(vector<Point2f>& p1, Mat& mask)
{
	vector<Point2f> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void maskout_colors(vector<Vec3b>& p1, Mat& mask)
{
	vector<Vec3b> p1_copy = p1;
	p1.clear();

	for (int i = 0; i < mask.rows; ++i)
	{
		if (mask.at<uchar>(i) > 0)
			p1.push_back(p1_copy[i]);
	}
}

void save_structure(string file_name, vector<Mat>& rotations, vector<Mat>& motions, Mat& structure, vector<Vec3b>& colors)
{
	int n = (int)rotations.size();

	FileStorage fs(file_name, FileStorage::WRITE);
	fs << "Camera Count" << n;
	fs << "Point Count" << structure.cols;

	fs << "Rotations" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << rotations[i];
	}
	fs << "]";

	fs << "Motions" << "[";
	for (size_t i = 0; i < n; ++i)
	{
		fs << motions[i];
	}
	fs << "]";

	fs << "Points" << "[";
	for (size_t i = 0; i < structure.cols; ++i)
	{
		Mat_<float> c = structure.col(i);
		c /= c(3);	//齐次坐标，需要除以最后一个元素才是真正的坐标值
		fs << Point3f(c(0), c(1), c(2));
	}
	fs << "]";

	fs << "Colors" << "[";
	for (size_t i = 0; i < colors.size(); ++i)
	{
		fs << colors[i];
	}
	fs << "]";

	fs.release();
}

int  main(int argc,char** argv)
{
	// if ( argc != 3 )
	//     {
	//         cout<<"usage: feature_extraction img1 img2"<<endl;
	//         return 1;
	//     }
	//     //-- 读取图像
	// Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
	// Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

	// Mat img_1 = imread ("1.png");
	// Mat img_2 = imread ("2.png");

	Mat img_1 = imread ("1.jpg");
	Mat img_2 = imread ("4.jpg");

	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;
	Mat descriptor1;
	Mat descriptor2;
	vector<Vec3b > colors1;
	vector<Vec3b > colors2;
	vector<DMatch> matches;
	//reading test
	// imshow("image1",img_1);
	// imshow("image2",img_2);
	// waitKey(0);

	//本征矩阵
	Mat K(Matx33d(
		1190.62, 0, 494.96,
		0, 1191.42, 1703.07,
		0, 0, 1));

	//提取特征
	extract_features(img_1,img_2, keypoints1, keypoints2 ,descriptor1, descriptor2,colors1,colors2);
	//特征匹配
	match_features(descriptor1, descriptor2, matches);

	//计算变换矩阵
	vector<Point2f> p1, p2;
	vector<Vec3b> c1, c2;
	Mat R, T;	//旋转矩阵和平移向量
	Mat mask;	//mask中大于零的点代表匹配点，等于零代表失配点
	get_matched_points(keypoints1, keypoints2, matches, p1, p2);
	get_matched_colors(colors1, colors2, matches, c1, c2);
	find_transform(K, p1, p2, R, T, mask);

	//三维重建
	Mat structure;	//4行N列的矩阵，每一列代表空间中的一个点（齐次坐标）
	maskout_points(p1, mask);
	maskout_points(p2, mask);
	//randu(R,Scalar::all(-500),Scalar::all(500));
	//cout<<"R(default)="<<endl<<R<<endl<<endl;
	reconstruct(K, R, T, p1, p2, structure);
	cout<<"R:"<<R<<endl;
	cout<<"t:"<<T<<endl;

	//保存并显示
	vector<Mat> rotations = { Mat::eye(3, 3, CV_64FC1), R };
	 vector<Mat> motions = { Mat::zeros(3, 1, CV_64FC1), T };
	maskout_colors(c1, mask);
	save_structure("structure.yml", rotations, motions, structure, c1);

	//system(".\\Viewer\\SfMViewer.exe");
	return 0;
}
