#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>  
#include "elas.h"
#include "image.h"

#include <time.h>

using namespace std;
using namespace cv;


#define THRE 2

double max(double a, double b)
{
	if (a < b)
		return b;
	else
		return a;
}

void mat2image(const cv::Mat& img, image<uchar>* &I){
	int width = img.cols;
	int height = img.rows;
	I = new image<uchar>(width, height);
	memcpy((char *)imPtr(I, 0, 0), img.datastart, width * height * sizeof(uchar));
}

void image2mat(const image<uchar>* I, cv::Mat& img){
	int width = I->width();
	int height = I->height();
	img.create(height, width, CV_8UC1);
	memcpy(img.datastart, (char *)imPtr(I, 0, 0), width * height * sizeof(uchar));
}

void computeDisparity(const image<uchar>* I1, const image<uchar>* I2, image<uchar>* &D1, image<uchar>* &D2){
	// get image width and height
	int32_t width = I1->width();
	int32_t height = I1->height();

	// allocate memory for disparity images
	const int32_t dims[3] = { width, height, width }; // bytes per line = width
	float* D1_data = (float*)malloc(width*height*sizeof(float));
	float* D2_data = (float*)malloc(width*height*sizeof(float));

	// process
	Elas::parameters param;
	param.postprocess_only_left = false;
	Elas elas(param);
	double t2 = (double)getTickCount();
	elas.process(I1->data, I2->data, D1_data, D2_data, dims);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();
	// find maximum disparity for scaling output disparity images to [0..255]
	float disp_max = 0;
	for (int32_t i = 0; i<width*height; i++) {
		if (D1_data[i]>disp_max) disp_max = D1_data[i];
		if (D2_data[i] > disp_max) disp_max = D2_data[i];
	}

	// copy float to uchar
	D1 = new image<uchar>(width, height);
	D2 = new image<uchar>(width, height);
	for (int32_t i = 0; i < width*height; i++) {
		D1->data[i] = (uint8_t)max(255.0*D1_data[i] / disp_max, 0.0);
		D2->data[i] = (uint8_t)max(255.0*D2_data[i] / disp_max, 0.0);
	}
}

void computeDisparity(const cv::Mat &I1, const cv::Mat &I2, cv::Mat &D1, cv::Mat &D2){
	// get image width and height
	int width = I1.cols;
	int height = I1.rows;

	// allocate memory for disparity images
	const int dims[3] = { width, height, width }; // bytes per line = width
	float* D1_data = (float*)malloc(width*height*sizeof(float));
	float* D2_data = (float*)malloc(width*height*sizeof(float));

	// process
	Elas::parameters param;
	param.postprocess_only_left = false;
	Elas elas(param);
	double t2 = (double)getTickCount();
	elas.process(I1.datastart, I2.datastart, D1_data, D2_data, dims);
	t2 = ((double)getTickCount() - t2) / getTickFrequency();
	// find maximum disparity for scaling output disparity images to [0..255]
	float disp_max = 0;
	for (int32_t i = 0; i<width*height; i++) {
		if (D1_data[i]>disp_max) disp_max = D1_data[i];
		if (D2_data[i]>disp_max) disp_max = D2_data[i];
	}

	D1 = cv::Mat::zeros(height, width, CV_32FC1);
	D2 = cv::Mat::zeros(height, width, CV_32FC1);
	for (int32_t i = 0; i < width*height; i++) {
		((float*)D1.data)[i] = D1_data[i];
		((float*)D2.data)[i] = D2_data[i];
	}
	// copy float to uchar
	/*D1 = cv::Mat::zeros(height, width, CV_8UC1);
	D2 = cv::Mat::zeros(height, width, CV_8UC1);
	for (int32_t i = 0; i < width*height; i++) {
		((uchar*)D1.data)[i] = (uint8_t)max(255.0*D1_data[i] / disp_max, 0.0);
		((uchar*)D2.data)[i] = (uint8_t)max(255.0*D2_data[i] / disp_max, 0.0);
	}*/
}

//compute v-disparity
image<uchar> * v_disparity(image<uchar> *disp, int threshold)
{
	int32_t width = disp->width();
	int32_t height = disp->height();
	float disp_max = 0;
	for (int i = 0; i < width*height; i++) {
		if (disp->data[i] > disp_max)
			disp_max = disp->data[i];
	}
	image<uchar> * vdisp = new image<uchar>(disp_max + 1, height);
	for (int i = 0; i < (disp_max + 1) * height; i++) {
		vdisp->data[i] = 0;
	}
	for (int i = 0; i <height; i++)
	{
		for (int j = 0; j<width; j++)
		{
			int value = disp->data[i*width + j];
			if (value > threshold)
			{
				int k = (disp_max + 1)*i + value;
				vdisp->data[k] += 1;
			}
		}
	}
	return vdisp;
}

Vec4i detectline(image<uchar> *disp, String outputPath, int ID)
{
	int width = disp->width();
	int height = disp->height();
	Mat midImage = Mat::zeros(height, width, CV_8U);
	//midImage.create(height,width,  CV_8U);
	//Mat midImage(width, height, CV_8U, Scalar::all(0));

	Mat BWImage;
	for (int r = 0; r < midImage.rows; r++)
	{
		for (int c = 0; c < midImage.cols; c++)
		{
			midImage.at<uchar>(r, c) = disp->data[r*width + c];
		}
	}

	Mat dstImage = Mat::zeros(height, width, CV_8UC3);
	for (int r = 0; r < midImage.rows; r++)
	{
		for (int c = 0; c < midImage.cols; c++)
		{
			dstImage.at<cv::Vec3b>(r, c)[0] = disp->data[r*width + c];
			dstImage.at<cv::Vec3b>(r, c)[1] = disp->data[r*width + c];
			dstImage.at<cv::Vec3b>(r, c)[2] = disp->data[r*width + c];
		}
	}

	int blockSize = 3;
	int constValue = 5;
	cv::adaptiveThreshold(midImage, BWImage, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY_INV, blockSize, constValue);

	for (int c = 0; c < BWImage.cols; c++)
	{
		long sumval = 0;
		for (int r = 0; r < BWImage.rows; r++)
		{
			sumval += BWImage.at<uchar>(r, c);
		}
		if (sumval>height/8 * 255)
			for (int r = 0; r < BWImage.rows; r++)
			{
				BWImage.at<uchar>(r, c) = 0;
			}
	}

	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合  
	HoughLinesP(BWImage, lines, 10, CV_PI / 180, 30, 50, 20);
	float len, max_len = 0;
	int temp;
	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		len = pow(l[0] - l[2], 2) + pow(l[1] - l[3], 2);
		if (len > max_len)
		{
			max_len = len;
			temp = i;
		}
	}
	//imshow("v",dstImage);
	line(dstImage, Point(lines[temp][0], lines[temp][1]), Point(lines[temp][2], lines[temp][3]), Scalar(0, 0, 255), 1, CV_AA);

	imshow("vdispline", dstImage);
	stringstream ss;
	ss << ID;
	//imwrite(outputPath + ss.str() + "_vdispline.png", dstImage);

	return lines[temp];

}






