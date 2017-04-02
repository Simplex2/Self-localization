#ifndef VPLANE_H
#define VPLANE_H

#include "_Header.h"

void ransac_fitline(const cv::Mat& vdisparity, cv::Vec4i& road_line);
Vec4i detectline_in_vdisparity(image<uchar> *disp, String outputPath, int ID);

void DetLane(Mat Img, Mat EdgeImg, Mat Disp, Mat Mask, Vec4i line, int *vp, int leftp[], int rightp[], cv::Mat &PtrX, cv::Mat &PtrY, String outputPath, int numOfFiles);
int DispWay(Mat ImgL, Mat ImgR, String outputPath, int ID, Vec4i &line, cv::Mat &Result, cv::Mat &Disp, cv::Mat &Mask);
void look4bestCtrPoints(Mat CostMap, int*vp, Mat disp, cv::Mat PtrX, cv::Mat PtrY, int*leftP, int*rightP);

#endif