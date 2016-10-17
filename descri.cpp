#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#include "descri.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

// constructor
descri::descri(string &imgPath)
{
	Mat input = imread(imgPath, -1); 
	imgToDes(input);
	//desToGrayImg(input);
}

// inputImg to des
void descri::imgToDes(Mat input)
{
	cout <<"not resort"<<endl;
	// resize input image
	Mat inputScaleTwo;
	//resize(input, inputScaleTwo, Size(input.cols*2, input.rows*2) );

	//to binary image with alpha value
	Mat alpha = alphaBinary(input);

	//color to Gray
	Mat inputGray;
	cvtColor(alpha, inputGray, CV_RGB2GRAY);

	//edge detection
	Mat inputCanny;
	Canny(inputGray, inputCanny, 50, 150, 3);
	
	// find contour
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(inputCanny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	Mat drawing = Mat::zeros( inputCanny.size(),CV_8UC1);

	//get sample points: points in vector
	sampleResult = getSamplePoints(contours[maxContour(contours)]);

	//get descriptor
	resultDescri = descriptor(sampleResult);
	//return grayDescri;
}

// alpha to binary
Mat descri::alphaBinary(Mat input)
{
	Mat alphaOrNot = Mat::zeros(input.size(),CV_8UC3);
	for(int i = 0 ; i < input.cols ; i++)
	{
		for(int j = 0 ; j < input.rows ; j++)
		{
			Vec4b & bgra = input.at<Vec4b>(j, i);
			Vec3b color = alphaOrNot.at<Vec3b>(j, i);
			if(bgra[3] != 0) // not transparency
			{
				color[0] = 255;
				color[1] = 255;
				color[2] = 255;
			}
			else
			{
				color[0] = 0;
				color[1] = 0;
				color[2] = 0;
			}
			alphaOrNot.at<Vec3b>(j, i) = color;
		}
	}
	return alphaOrNot;
}

//return index of max contour
int descri::maxContour(vector<vector<Point>> contours)
{
	int maxSize = -1;
	int maxIndex = -1;
	for(int i = 0 ; i < contours.size() ; i++)
	{
		
		if(int(contours[i].size()) > maxSize)
		{
			//cout << "index: "<<i<<", size: "<<contours[i].size() << endl;
			maxSize = int(contours[i].size());
			maxIndex = i;
		}
	}
	cout << "maxsize: "<<maxSize<<endl;
	return maxIndex;
}

// return contour length
float descri::contourLength(vector<Point> singleContour)
{
	float totalDist = 0;
	Point start;
	Point finish;
	start = singleContour[0];

	for(int i = 1 ; i < singleContour.size() ; i++)
	{
		finish = singleContour[i];
		cout << sqrt(pow(finish.x-start.x,2)+pow(finish.y-start.y,2)) << endl;
		totalDist += sqrt(pow(finish.x-start.x,2)+pow(finish.y-start.y,2));
		//cout << "x: "<<singleContour[i].x<<endl;
		//cout << "y: "<<singleContour[i].y<<endl;
		start = finish;
	}

	return totalDist;
}

// return vector of sample point
vector<Point> descri::getSamplePoints(vector<Point> singleContour)
{
	int pointCount = 50;
	int totalPoints = singleContour.size();

	vector<int> pointIndex;
	//vector<Point> samplePoints;
	int count = singleContour.size();
	int tmp = 0;

	//float step = totalPoints/pointCount;

	//cout <<"step: " <<step<<endl;

	for(int i = 0 ; i < pointCount ; i++)
	{

		tmp = 0 + i*totalPoints/pointCount;
		//cout << tmp << " ";
		pointIndex.push_back(tmp);
	}
	//cout << endl;
	sort(pointIndex.begin(),pointIndex.end());

	for(int i = 0 ; i < pointIndex.size() ; i++)
	{
		sampleResult.push_back(singleContour[pointIndex[i]]);
	}

	return sampleResult;
}


// get descriptor
Mat descri::descriptor(vector<Point> samplePoints)
{
	Point pi;
	Point pj;
	Point pjMinusDelta;
	
	int delta = 3;
	float tmp = 0;
	Mat shapeDes = Mat::zeros(samplePoints.size()/*+samplePoints.size()/2*/,samplePoints.size()/*+samplePoints.size()/2*/,CV_32FC1); 
	for(int i = 0 ; i < shapeDes.rows ; i++)
	{
		for(int j = 0 ; j < shapeDes.cols ; j++)
		{
			pi = samplePoints[i/* % samplePoints.size()*/];
			pj = samplePoints[j/* % samplePoints.size()*/];
			
			
			if(abs(int((i/* % samplePoints.size()*/)-(j/* % samplePoints.size()*/))) < delta)
			{
				shapeDes.at<float>(i,i) = 0;
			}
			else 
			{
				if((i/* % samplePoints.size()*/) > (j/* % samplePoints.size()*/))
					pjMinusDelta = samplePoints[((j/* % samplePoints.size()*/)+delta/*+samplePoints.size()*/)/*%samplePoints.size()*/];
				else
					pjMinusDelta = samplePoints[((j/* % samplePoints.size()*/)-delta/*+samplePoints.size()*/)/*%samplePoints.size()*/];

				tmp = angle(pi, pj, pjMinusDelta);
				//cout <<"i: "<<i<<", j:" <<j<<", angle: "<<255-(tmp*255/180) <<endl;
				//shapeDes.at<float>(i,j) = (tmp*255/180);
				shapeDes.at<float>(i,j) = tmp;
				//shapeDes.at<float>(j,i) = 255-(tmp*255/180);
			}
		}
	}
	return shapeDes;
}

// normalize descriptor into 0~255

//void descri::desToGrayImg(Mat input)
//{
//	// resize input image
//	Mat inputScaleTwo;
//	resize(input, inputScaleTwo, Size(input.cols*2, input.rows*2) );
//
//	//to binary image with alpha value
//	Mat alpha = alphaBinary(inputScaleTwo);
//
//	//color to Gray
//	Mat inputGray;
//	cvtColor(alpha, inputGray, CV_RGB2GRAY);
//
//	//edge detection
//	Mat inputCanny;
//	Canny(inputGray, inputCanny, 50, 150, 3);
//	
//	// find contour
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//	findContours(inputCanny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
//	Mat drawing = Mat::zeros( inputCanny.size(),CV_8UC1);
//
//	//get sample points: points in vector
//	vector<Point>tmpGG = getSamplePoints(contours[maxContour(contours)]);
//
//	//get descriptor
//	Mat tmpGeneralDescri = descriptor(tmpGG);
//
//	Mat grayDescri = Mat::zeros( tmpGeneralDescri.rows, tmpGeneralDescri.cols ,CV_32FC1);
//
//	for(int i = 0 ; i < grayDescri.rows ; i++)
//	{
//		for(int j = 0 ; j < grayDescri.cols ; j++)
//		{
//			grayDescri.at<float>(i,j) = (tmpGeneralDescri.at<float>(i,j)*255/180);
//		}
//	}
//}

// get angle
float descri::angle(Point i, Point j, Point jMinusDelta)
{
	//A = i, B = j, C = jMinusDelta
	float cosAngle = 0; //corner i_j_jMinusDelta
	float angle = 0;
	float distA = 0; //BC
	float distB = 0; //AC
	float distC = 0; //AB

	distA = sqrt(pow(j.x-jMinusDelta.x, 2) + pow(j.y-jMinusDelta.y, 2));
	distB = sqrt(pow(i.x-jMinusDelta.x, 2) + pow(i.y-jMinusDelta.y, 2));
	distC = sqrt(pow(i.x-j.x, 2) + pow(i.y-j.y, 2));

	if(distA == 0 || distC == 0)
		angle = 0;
	else
	{
		cosAngle = (pow(distA,2)+pow(distC,2)-pow(distB,2))/(2*distC*distA);
		if(cosAngle>1)
			angle = 0;
		else if(cosAngle<-1)
			angle = 180;
		else
			angle = acos(cosAngle) * 180.0 / PI;
	}
	
	return angle;
}