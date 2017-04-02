#ifndef DISPARITY_H
#define DISPARITY_H
#include <opencv2/opencv.hpp>  
#include "image.h"


using namespace std;
using namespace cv;



//namespace disparity{
	double max(double a, double b);
	void mat2image(const cv::Mat& img, image<uchar>* &I);
	void image2mat(const image<uchar>* I, cv::Mat& img);
	void computeDisparity(const image<uchar>* I1, const image<uchar>* I2, image<uchar>* &D1, image<uchar>* &D2);
	void computeDisparity(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &D1, cv::Mat &D2);

	image<uchar> * v_disparity(image<uchar> *disp, int threshold);
	Vec4i detectline(image<uchar> *disp, String outputPath, int ID);

//};

#endif