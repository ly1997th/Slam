#ifndef CAMERA_H
#define CAMERA_H
//if the name is not defined, then define the mane.

#include "myslam/common_include.h"

namespace myslam
{
	//pinhole RGB_D camera model

	class Camera
	{
	public:
		typedef std::shared_ptr<Camera> Ptr;
		float fx_,fy_,cx_,cy_,depth_scale_;//Camera intrinsics

		Camera();

		Camera(float fx,float fy,float cx,float cy,float depth_scale=0):fx_(fx),fy_(fy),cx_(cx),cy_(cy),depth_scale(depth_scale_){}

		//coordinate transform:world,camera,pixel
		Vector3d world_to_camera(const Vector3d &p_w, const SE3& T_c_w);
		Vector3d camera_to_world(const Vector3d &p_c, const SE3& T_c_w);
		Vector2d camera_to_pixel(const Vector3d &p_c);
		Vector3d pixel_to_camera(const Vector2d &p_p,double depth=1);//the normalized plane
		Vector3d pixel_to_world(const Vector2d &p_p,const SE3 &T_c_w,double depth=1);
		Vector2d world_to_pixel(const Vector3d &p_w,const SE3 &T_c_w);
	};
}

#endif //CAMERA_H