#include "myslam/frame.h"

namespace myslam
{
	Frame::Frame():id_(-1),time_stamp_(-1),camera_(nullptr){}
	Frame::Frame(long id, double time_stamp, SE3 T_c_w, Camera ::Ptr camera,Mat color,Mat depth)
	{

	}

	Frame::~Frame()
	{

	}

	Frame::Ptr Frane::createFrame()
	{
		static long factory_id = 0;
		return Frame::Ptr(new Frame(Factory_id++));
	}

	double Frame::findDepth(const cv::KeyPoint &kp)
	{
		int x=cvRound(kp.pt.x);
		int y=cvRound(kp.pt.y);
		ushort d=depth_.ptr<ushort>(y)[x];
		if(d!=0)
		{
			return double(d)/camera_->depth_scale_;
		}
		else
		{
			int dx[4]={-1,0,1,0};
			int dy[4]={0,-1,0,1};
			for (int i=0;i<4;i++)
			{
				d=depth_.ptr<ushort>(y+dy[i])[x+dx[i]];
				if(d!=0)
				{
					return double(d)/camera_>depth_scale_;
				}
			}
		}
		return -1.0;
	}
}