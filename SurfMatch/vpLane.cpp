#include "vpLane.h"
//#include "ransacLine.h"
int DispWay(Mat ImgL, Mat ImgR, String outputPath, int ID, Vec4i &line, cv::Mat &Result, cv::Mat &Disp, cv::Mat &Mask)
{
	int Height = ImgL.rows;
	int Width = ImgL.cols;

	Mat Obst;
	int hl;
	ImgL.copyTo(Obst);
	ImgL.copyTo(Result);
	cvtColor(ImgL, ImgL, CV_BGR2GRAY);
	cvtColor(ImgR, ImgR, CV_BGR2GRAY);
	image<uchar> *I1 = nullptr;
	mat2image(ImgL, I1);
	image<uchar> *I2 = nullptr;
	mat2image(ImgR, I2);
	// process
	image<uchar> *D1;
	image<uchar> *D2;
	computeDisparity(I1, I2, D1, D2);
	//Mat imgDisp1, imgDisp2;
	//computeDisparity(ImgL, ImgR, imgDisp1, imgDisp2);
	image<uchar> *vdipa;
	int threshold = 5;
	vdipa = v_disparity(D1, threshold);
	Mat	VDisp;
	VDisp.create(vdipa->height(), vdipa->width(), CV_8UC1);
	for (int i = 0; i < vdipa->height(); i++)
		for (int j = 0; j < vdipa->width(); j++)
			VDisp.at<uchar>(i, j) = vdipa->data[i*vdipa->width() + j];
	//imshow("VDisp", VDisp);
	//operate_vdisparity(VDisp, line);
	//cv::line(VDisp, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 0, 255), 1, CV_AA);
	//imshow("VDisp",VDisp);
	/*line = detectline(vdipa, outputPath, ID);*/
	line = detectline_in_vdisparity(vdipa, outputPath, ID); //by ransac
	float t = (line[3] - line[1]);
	float k = (line[2] - line[0]) / t;
	float b = line[0] - k*line[1];
	hl = -b / k;

	for (int j = 0; j < Width; j++)
	{
		Result.at<cv::Vec3b>(hl, j)[2] = 255;
	}
	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			int d_value = D1->data[i*Width + j];
			Disp.at<uchar>(i, j) = d_value;
			int v_value = k*i + b;
			//if (abs(d_value - v_value)<5 + v_value / 20)
			if (abs(d_value - v_value) < (float)v_value*0.2)
			{
				Result.at<cv::Vec3b>(i, j)[2] = 255;
			}
			if (abs(d_value - v_value) > (float)v_value*0.4 && d_value)
			{
				Mask.at<uchar>(i, j) = 0;
				Obst.at<cv::Vec3b>(i, j)[1] = 0;
				//Obst.at<cv::Vec3b>(i, j)[2] = 255;
			}
			else
			{
				Mask.at<uchar>(i, j) = 1;
			}

		}
	}
	stringstream ss;
	ss << ID;
	//imshow("Disp", Disp);
	imwrite(outputPath + ss.str() + "_Disp.png", Disp);

	imwrite(outputPath + "disparity\\" + ss.str() + "_Disp.png", Disp);
	//imshow("Obst", Obst);

	//imwrite(outputPath + ss.str() + "_DispRoad.png", Result);
	return hl;
}