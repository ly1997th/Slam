#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace chrono;

int main(int argc,char **argv)
{
	if(argc!=2)
	{
		cout<<"usage:useLK path_to_dataSet"<<endl;
		return 1;
	}

	string path_to_dataset=argv[1];
	string associate_file=path_to_dataset+"/associate.txt";

	ifstream fin(associate_file);
	if ( !fin ) 
    {
        cerr<<"I cann't find associate.txt!"<<endl;
        return 1;
    }
    
	string rgb_file,depth_file,time_rgb,time_depth;
	list<cv::Point2f>keypoints;
	cv::Mat color,depth,last_color;

	for(int index=0;index<100;index++)
	{
		fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
		color=cv::imread(path_to_dataset+"/"+rgb_file);
		depth=cv::imread(path_to_dataset+"/"+depth_file,-1);
		if(index==0)
		{
			vector<cv::KeyPoint> kps;
			cv::Ptr<cv::FastFeatureDetector> detector=cv::FastFeatureDetector::create();
			detector->detect(color,kps);
			for(auto kp:kps)
				keypoints.push_back(kp.pt);
			last_color=color;
			continue;
		}
		if(color.data==nullptr||depth.data==nullptr)
			continue;
		vector<cv::Point2f> prev_keypoints,next_keypoints;
		for(auto kp:keypoints)
			prev_keypoints.push_back(kp);
		vector<unsigned char> status;
		vector<float> error;

		steady_clock::time_point t1=steady_clock::now();
		calcOpticalFlowPyrLK(last_color,color,prev_keypoints,next_keypoints,status,error);
		steady_clock::time_point t2=steady_clock::now();
		duration<double> time_used=duration_cast<duration<double>>(t2-t1);
		cout<<"LK Flow use time:"<<time_used.count()<<"seconds."<<endl;

		int i=0;
		for(auto iter=keypoints.begin();iter!=keypoints.end();i++)
		{
			if(status[i]==0)
			{
				iter=keypoints.erase(iter);
				continue;
			}
			*iter=next_keypoints[i];
			iter++;
		}
		cout<<"tracked keypoits:"<<keypoints.size()<<endl;
		if(keypoints.size()==0)
		{
			cout<<"all keypoints aree lost."<<endl;
			break;
		}
		cv::Mat img_show=color.clone();
		for(auto kp:keypoints)
			cv::circle(img_show,kp,10,cv::Scalar(0,240,0),1);
		cv::imshow("corners",img_show);
		cv::waitKey(0);
		last_color=color;
	}
	return 0;
}