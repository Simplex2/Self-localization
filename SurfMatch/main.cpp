#include "_Header.h"

const string inputPath = ".\\pics\\";
const string outputPath = ".\\output\\";

int main()
{
	//time_t bg, ed;//计时用
	
	int Height = 188;
	int Width = 620;
	Mat P1 = (Mat_<double>(3, 4) << 718.8560, 0, 607.1928, 0, 0, 718.8560, 185.2157, 0, 0, 0, 1, 0);
	Mat P2 = (Mat_<double>(3, 4) << 718.8560, 0, 607.1928, -386.1448, 0, 718.8560, 185.2157, 0, 0, 0, 1, 0);

	for (int id = 0; id < 20; id++)
	{
		//这么定义disparity会计算了两次，之后可以更改使得前一次计算的disparity可以给下一次使用         ！！！！！
		Mat colorImgL, colorImgR;            //t时刻图像
		Mat colorImgNL,colorImgNR;           //t+1时刻图像
		Mat Disp_L, Disp_R;                //t时刻的disparity
		Mat Disp_NL, Disp_NR;                //t+1时刻的disparity
		//Mat Depth_L, Depth_NL;               //深度

		stringstream ss,ssl;
		ss << id;                    //等于cin>>id to ss
		colorImgL = imread(inputPath + "left\\" + ss.str() + ".png");
		resize(colorImgL, colorImgL, Size(Width, Height));
		colorImgR = imread(inputPath + "right\\" + ss.str() + ".png");
		resize(colorImgR, colorImgR, Size(Width, Height));
		int nextid = id + 1;
		ssl << nextid;
		colorImgNL = imread(inputPath + "left\\" + ssl.str() + ".png");
		resize(colorImgNL, colorImgNL, Size(Width, Height));
		colorImgNR = imread(inputPath + "right\\" + ssl.str() + ".png");
		resize(colorImgNR, colorImgNR, Size(Width, Height));

		//把原图转为灰度图
		cvtColor(colorImgL, colorImgL, CV_BGR2GRAY);
		cvtColor(colorImgR, colorImgR, CV_BGR2GRAY);
		cvtColor(colorImgNL, colorImgNL, CV_BGR2GRAY);
		cvtColor(colorImgNR, colorImgNR, CV_BGR2GRAY);

		
		//Compute disparity
		computeDisparity(colorImgL, colorImgR, Disp_L, Disp_R);
		computeDisparity(colorImgNL, colorImgNR, Disp_NL, Disp_NR);

		//imshow("ds",Disp_L);

		//Compute Depth

		//reprojectImageTo3D(Disp_L, Depth_L,Camera_param);
		////cvtColor(Depth_L, Depth_L, CV_BGR2GRAY);
		//imshow("depth",Depth_L);
		//cout << Depth_L.rows<<" "<<Depth_L.cols<<endl;

		//Extract feature
		int minHessian = 400;
		SurfFeatureDetector detector(minHessian);
		vector<KeyPoint> keypoint_1, keypoint_2;

		detector.detect(colorImgL, keypoint_1);
		detector.detect(colorImgNL, keypoint_2);
		//Mat img_keypoints_1;
		//drawKeypoints(colorImgL, keypoint_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		//imshow("f", img_keypoints_1);

		//将vector向量变成向量矩阵形式保存在Mat中
		SurfDescriptorExtractor extractor;
		Mat descriptors1, descriptors2;
		extractor.compute(colorImgL, keypoint_1, descriptors1);                            
		extractor.compute(colorImgNL, keypoint_2, descriptors2);
		
		//利用BruteFroceMatch对保存在Mat中的描述子进行匹配
		BruteForceMatcher<L2 <float>>matcher;
		vector<DMatch>matches;                             //保存所有匹配上的特征点的index：queryIdx，trainIdx
		matcher.match(descriptors1, descriptors2, matches);
		//Mat imgMatches;
		//drawMatches(colorImgL, keypoint_1, colorImgNL, keypoint_2, matches, imgMatches);
		//imshow("m", imgMatches);

		
		//用两个Mat保存所有匹配点所对应的坐标及深度
		int num_match = matches.size();
		Mat matchpoint_imgL(num_match, 4, CV_64FC1);
		Mat matchpoint_imgNL(num_match, 4, CV_64FC1);

		Mat realword_pointL(num_match, 4, CV_64FC1);
		Mat realword_pointNL(num_match, 4, CV_64FC1);
		Mat A = Mat::zeros(4, 4, CV_64FC1);
		Mat w, u, vt;
		A.copyTo(w);
		A.copyTo(u);
		A.copyTo(vt);

		for (int id_match = 0; id_match < num_match; id_match++)
		{
			float id_keypoint1 = matches[id_match].queryIdx;
			float xL_match = keypoint_1[id_keypoint1].pt.x;
			float yL_match = keypoint_1[id_keypoint1].pt.y;

		/*	if (int(Disp_L.at<uchar>(yL_match, xL_match))>100)
			{
				Disp_L.at<uchar>(yL_match, xL_match) = 300;
			}*/


			float xR_match = xL_match - Disp_L.at<float>(yL_match, xL_match);
			float yR_match = yL_match;
			A.row(0) = (xL_match*P1.row(2) - P1.row(0));
			A.row(1) = (yL_match*P1.row(2) - P1.row(1));
			A.row(2) = (xR_match*P2.row(2) - P2.row(0));
			A.row(3) = (yR_match*P2.row(2) - P2.row(1));

			SVD::compute(A, w, u, vt);
			Mat x = vt.row(3);
			x = x / x.at<double>(3);
		
			x.copyTo(realword_pointL.row(id_match));
			//if (id_match == 186)
			//{
			//	cout << "A: " << A << endl;
			//	cout << xL_match << " " << yL_match << " " << int(Disp_L.at<float>(yL_match, xL_match)) << endl;
			//	cout << "vt: " << vt << endl;
			//	cout << "x: " << x << endl;
			//	
			//}
		}

		Mat point_previous = realword_pointL.colRange(0, 3).clone();
		

		for (int id_match = 0; id_match < num_match; id_match++)
		{
			float id_keypoint2 = matches[id_match].trainIdx;
			float xNL_match = keypoint_2[id_keypoint2].pt.x;
			float yNL_match = keypoint_2[id_keypoint2].pt.y;

			//if (int(Disp_L.at<uchar>(yNL_match, xNL_match))>100)
			//{
			//	Disp_L.at<uchar>(yNL_match, xNL_match) = 600;
			//}

			float xNR_match = xNL_match - Disp_L.at<float>(yNL_match, xNL_match);
			float yNR_match = yNL_match;

			A.row(0) = (xNL_match*P1.row(2) - P1.row(0));
			A.row(1) = (yNL_match*P1.row(2) - P1.row(1));
			A.row(2) = (xNR_match*P2.row(2) - P2.row(0));
			A.row(3) = (yNR_match*P2.row(2) - P2.row(1));
			SVD::compute(A, w, u, vt);
			Mat x = vt.row(3);
			x = x / x.at<double>(3);
			x.copyTo(realword_pointNL.row(id_match));
			//if (id_match == 186)
			//{
			//	cout << "A: " << A << endl;
			//	cout << xNL_match << " " << yNL_match << " " << int(Disp_L.at<float>(yNL_match, xNL_match)) << endl;
			//	cout << "vt: " << vt << endl;
			//	cout << "x: " << x << endl;
			//}
		}

		Mat point_next = realword_pointNL.colRange(0, 3).clone();
		
		//建立consistent matrix
		Mat consistent_mat = Mat::zeros(num_match, num_match, IPL_DEPTH_1U);
		int consistent_count = 0;
		int abs_max=0;//保存连结最多节点的节点个数
		int max_node=0;//保存联结最多节点的index
		//cout << consistent_count << "   ";
		for (int i = 0; i < num_match; ++i)
		{
			int edge_count = 0;
			for (int j = 0; j < num_match; ++j)
			{
				//cout << point_next.row(i) << "   " << point_next.row(j) << "   " << point_previous.row(i)<<"   " << point_previous.row(j)<<endl;
				if (abs(norm(point_next.row(i), point_next.row(j)) - norm(point_previous.row(i), point_previous.row(j))) < 0.02)
				{
					consistent_mat.at<uchar>(i, j) = 1;
					edge_count++;
					consistent_count++;
				}
			}
			if (edge_count > abs_max)
			{
				abs_max = edge_count;
				max_node = i;
			}
		}
		//cout << consistent_mat << endl;
		cout << consistent_count <<" max node: "<<max_node<<" connected nodes: "<<abs_max<< endl;

		//Cacluate maximum clique
		vector<int> clique{ max_node };
		int curr_node = max_node;
		int curr_max = abs_max;
		while (curr_max > 0)
		{
			int sub_max = 0;
			int sub_node = 0;
			for (int i = 0; i < num_match; i++)
			{
				vector<int>::iterator it1 = find(clique.begin(), clique.end(), i);
				if (it1!=clique.end())
					continue;
				else
				{
					if (consistent_mat.at<uchar>(curr_node, i) == 1)
					{
						int sub_count = 0;
						for (int j = 0; j < num_match; j++)
						{
							vector<int>::iterator it2 = find(clique.begin(), clique.end(), j);
							if (it2 != clique.end())
								continue;
							else
							{
								if (consistent_mat.at<char>(i, j) == 1)
								{
									bool cotain=1;
									for (int last = 0; last < clique.size(); last++)
									{
										if (consistent_mat.at<char>(j, clique[last]) != 1)
											cotain = 0;
									}
									if (cotain==1)
									sub_count++;
								}
								if (sub_count > sub_max)
								{
									sub_max = sub_count;
									sub_node = j;
								}
							}
						}
					}
				}
				
			}
			curr_max = sub_max;
			curr_node = sub_node;
			if (sub_node!=0)
			clique.push_back(sub_node);
			//cout <<sub_node<<" "<< sub_max << " ";
		}
		cout << "clique Size: "<<clique.size()<<endl;
		for (vector<int>::const_iterator it = clique.cbegin(); it != clique.end();it++)
		{
			cout << (*it) << " " << point_previous.at<double>(*it, 0) << " " << point_previous.at<double>(*it, 1) << " " << point_previous.at<double>(*it, 2) << endl;
			cout << "NL: " << point_next.at<double>(*it, 0) << " " << point_next.at<double>(*it, 1) << " " << point_next.at<double>(*it, 2) << endl;
			//cout << *it << " ";
			cout << endl;
		}
		cout << endl;
		waitKey(1);
	}
	return 0;

}
