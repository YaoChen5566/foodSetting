#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#include "compare.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

// constructor
comp::comp(Mat descri1, Mat descri2)
{
	compareDes(descri1, descri2);
}

void comp::compareDes(Mat input1, Mat input2)
{
	
	int r = 25; // square size
	int lefttopPoint1 = 0;
	int lefttopPoint2 = 0;
	float currentScore = 10000000;
	
	vector <double> subSum1;
	vector <double> subSum2;
	
	double getScore = 0;
	Mat desInteg1;
	Mat desInteg2;

	double tmpSum1 = 0;
	double tmpSum2 = 0;

	integral(input1, desInteg1);
	integral(input2, desInteg2);

	for(int i = 0 ; i < input1.cols ; i++)
	{
		if( (i+r) <= input1.cols)
		{
			tmpSum1 = desInteg1.at<double>(i, i) + desInteg1.at<double>(i+r, i+r)-desInteg1.at<double>(i, i+r)-desInteg1.at<double>(i+r, i);
			
			tmpSum2 = desInteg2.at<double>(i, i) + desInteg2.at<double>(i+r, i+r)-desInteg2.at<double>(i, i+r)-desInteg2.at<double>(i+r, i);
		}
		else
		{
			tmpSum1 += desInteg1.at<double>(i+r-input1.cols+1, i+r-input1.cols+1); //[y, y]
			tmpSum1 += desInteg1.at<double>(input1.cols-1, input1.cols-1)+desInteg1.at<double>(i, i)-desInteg1.at<double>(i, input1.cols-1)-desInteg1.at<double>(input1.cols-1, i); //[x, x]
			tmpSum1 += desInteg1.at<double>(i+r-input1.cols+1, input1.cols-1)-desInteg1.at<double>(i+r-input1.cols+1, i); //[y, cols]
			tmpSum1 += desInteg1.at<double>(input1.cols-1, i+r-input1.cols+1)-desInteg1.at<double>(i, i+r-input1.cols+1); //[cols, y]

			tmpSum2 += desInteg2.at<double>(i+r-input1.cols+1, i+r-input1.cols+1); //[y, y]
			tmpSum2 += desInteg2.at<double>(input1.cols-1, input1.cols-1)+desInteg2.at<double>(i, i)-desInteg2.at<double>(i, input1.cols-1)-desInteg2.at<double>(input1.cols-1, i); //[x, x]
			tmpSum2 += desInteg2.at<double>(i+r-input1.cols+1, input1.cols-1)-desInteg2.at<double>(i+r-input1.cols+1, i); //[y, cols]
			tmpSum2 += desInteg2.at<double>(input1.cols-1, i+r-input1.cols+1)-desInteg2.at<double>(i, i+r-input1.cols+1); //[cols, y]
		}
		subSum1.push_back(tmpSum1);
		subSum2.push_back(tmpSum2);
		
		tmpSum1 = 0;
		tmpSum2 = 0;
	}

	for(int i = 0 ; i < subSum1.size() ; i++)
	{
		for(int j = 0 ; j < subSum2.size() ; j++)
		{
			getScore = subSum1[i] - subSum2[j];
			getScore /= pow(r, 2);
			if(getScore < currentScore)
			{
				 
				currentScore = getScore;
				startIndex1 = i;
				startIndex2 = j;
				range = r;
			}
		}
	}

	/*
	for(int i = 0 ; i < input1.cols ; i++)
	{
		for(int j = 0 ; j < input2.cols ; j++)
		{
			Mat tmp1 = subMatrix(input1, i, i, r);
			Mat tmp2 = subMatrix(input2, j, j, r);

			Mat minus;
			absdiff(tmp1, tmp2, minus);
			
			getScore = cv::sum(minus)[0];
				
			getScore /= pow(r,2);

			if(getScore < currentScore)
			{
				 
				currentScore = getScore;
				startIndex1 = i;
				startIndex2 = j;
				range = r;
			}
		}
	}
	score = currentScore;
	cout << "score: "<<currentScore<<endl;
	*/
}

Mat comp::subMatrix(Mat input, int row, int col, int range)
{
	Mat subM = Mat::zeros(range, range, CV_32FC1);
	for(int i = 0 ; i < range ; i++)
	{
		for(int j = 0 ; j < range ; j++)
		{
			subM.at<float>(i, j) = input.at<float>((i+row)%input.rows, (j+col)%input.cols);
		}
	}
	return subM;
}

