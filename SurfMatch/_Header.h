#ifndef _HEADER_H
#define _HEADER_H

#include <iostream>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>  
#include<opencv2/highgui/highgui.hpp>  
#include<opencv2/imgproc/imgproc.hpp>  
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
//#include "livewire.h"
//#include "fLiveWireCalcP.h"
#include "disparity.h"
#include "elas.h"
#include "image.h"
//#include "ransacLine.h"
#include "vpLane.h"

using namespace std;
using namespace cv;







struct coor
{
	int x;
	int y;
};



#endif