#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include "harris.h"

using namespace std;
using namespace cv;


int main(int argc,char** argv)
{
   Mat  image_origin,image;
   //image_origin1= cv::imread (argv[1]);  
   image_origin= cv::imread ("1.jpg");
    if ( image_origin.data == nullptr ) //数据不存在,可能是文件不存在    
    {
        cerr<<"文件没有正确输入."<<endl;
        return 0;
    }
   cvtColor (image_origin,image,CV_BGR2GRAY);  
  
  
   // 经典的harris角点方法  
   harris Harris;  
   // 计算角点  
   Harris.detect(image);  
    

   //获得角点  
   vector<Point> pts;  
   Harris.getCorners(pts,0.01);  
   // 标记角点  
   Harris.drawOnImage(image,pts);  
   cv::namedWindow ("harris");  
   cv::imshow ("harris",image);  
   cv::waitKey (0); 
   imwrite("harris1.jpg", image);
   
   // cv::namedWindow ("harrisResponse");  
   // cv::imshow ("harrisResponse",Harris.harrisResponse);  //harrisResponse
   // cv::waitKey (0);
   cv::namedWindow ("harrisResponse");  
   cv::imshow ("harrisResponse",Harris.harrisResponse1);  //harrisResponse
   cv::waitKey (0);
   imwrite("harrisResponse1.jpg",Harris.harrisResponse1);
   
   cv::namedWindow ("thresh");  
   cv::imshow ("thresh",Harris.thresh);  
   cv::waitKey (0); 
    imwrite("thresh1.jpg", Harris.thresh);

   cv::namedWindow ("localmax");  
   cv::imshow ("localmax",Harris.localmax);  
   cv::waitKey (0); 
    imwrite("localmax1.jpg",Harris.localmax);
   
   return 0;  
}