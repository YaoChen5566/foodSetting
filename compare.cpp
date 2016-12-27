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
#include "fragment.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

// constructor
comp::comp()
{
	setInitial();
}

comp::comp(Mat descri1, Mat descri2)
{
	setInitial();
	compareDes(descri1, descri2);
}

comp::comp(Mat descri1, vector<Mat> descri2Seq)
{
	setInitial();
	for(int i = 0 ; i < descri2Seq.size() ; i++)
		compareDesN(descri1, descri2Seq[i], i);
}

comp::comp(Mat descri1, vector<Mat> descri2Seq, vector<Point> pointSeq1, vector<Point> pointSeq2)
{
	setInitial();
	_pointSeq1 = pointSeq1;
	_pointSeq2 = pointSeq2;
	for(int i = 0 ; i < descri2Seq.size() ; i++)
		compareDesN(descri1, descri2Seq[i], i);
}

//set initial
void comp::setInitial()
{
	_thresholdScore = 1.0;
	_startIndex1 = 0;
	_startIndex2 = 0;
	_range = 0;
}

//two single image
void comp::compareDes(Mat input1, Mat input2)
{
	
	int rLim = 5; // square size
	int lefttopPoint1 = 0;
	int lefttopPoint2 = 0;
	//float currentScore = 0;
	
	vector <double> subSum1;
	vector <double> subSum2;
	
	double getScore = 0;
	Mat desInteg1;
	Mat desInteg2;

	double tmpSum1 = 0;
	double tmpSum2 = 0;

	integral(input1, desInteg1);
	integral(input2, desInteg2);

	for(int r = rLim ; r < input1.cols/3 ; r++)
	{
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
			
				getScore = pow(abs(subSum1[i] - subSum2[j]), 2);
				getScore /= pow(r, 2);
				if(getScore < _score)
				{
				 
					//currentScore = getScore;
					_startIndex1 = i;
					_startIndex2 = j;
					_range = r;
					_score = getScore;
				}
			}
		}
		
		subSum1.clear();
		subSum2.clear();
	}
}

//single and a n seq
void comp::compareDesN(Mat input1, Mat input2, int index)
{
	Mat sub = input1-input2;
	Mat integral1; // sum
	Mat integral2; // square sum

	int rLim = 5; // square size
	int lefttopPoint1 = 0;
	int lefttopPoint2 = 0;
	double tmpSum = 0;
	double getScore = 0;
	integral(sub, integral1, integral2);
	


	for(int r = input1.cols*0.8 ; r > rLim ; r--)
	{
		for(int i = 0 ; i < input1.cols ; i++)
		{
			if( (i+r) <= input1.cols)
			{
				tmpSum = integral2.at<double>(i, i) + integral2.at<double>(i+r, i+r)-integral2.at<double>(i, i+r)-integral2.at<double>(i+r, i);
			
			}
			else
			{
				tmpSum += integral2.at<double>(i+r-input1.cols+1, i+r-input1.cols+1); //[y, y]
				tmpSum += integral2.at<double>(input1.cols-1, input1.cols-1)+integral2.at<double>(i, i)-integral2.at<double>(i, input1.cols-1)-integral2.at<double>(input1.cols-1, i); //[x, x]
				tmpSum += integral2.at<double>(i+r-input1.cols+1, input1.cols-1)-integral2.at<double>(i+r-input1.cols+1, i); //[y, cols]
				tmpSum += integral2.at<double>(input1.cols-1, i+r-input1.cols+1)-integral2.at<double>(i, i+r-input1.cols+1); //[cols, y]
			}

			getScore = tmpSum/pow(r,2);

			if(getScore < _thresholdScore)
			{
				_range = r;
				_score = getScore;
				_startIndex1 = i;
				_startIndex2 = i+index;

				// calculate the warping matrix
				vector<Point> matchSeqR = subPointSeq(_pointSeq1, _startIndex1, _range);
				vector<Point> matchSeqQ = subPointSeq(_pointSeq2, _startIndex2, _range);

				Mat warpMat = estimateRigidTransform(matchSeqQ, matchSeqR, false); // (src/query, dst/reference)



				// use scale and whether generate the warping matrix to judge the fragment probablity
				if(warpMat.size() != cv::Size(0, 0))
				{
					double scale = pow(warpMat.at<double>(0, 0), 2) + pow(warpMat.at<double>(1, 0), 2);
					if (abs(scale-1.0) < 0.1)
					{
						frag fragment(i, i+index, r);
					
						if(!fragExist(fragment))
							_frag.push_back(fragment);
					}
				}
			}
		}
	}


}

//range
int comp::range()
{
	return _range;
}

//start index 1
int comp::startIndex1()
{
	return _startIndex1;
}

//start index 2
int comp::startIndex2()
{
	return _startIndex2;
}

//score
double comp::score()
{
	return _score;
}

// return a set of fragment
vector<frag> comp::fragList()
{
	return _frag;
}

// check if the same frag in the vector
bool comp::fragExist(frag newFrag)
{
	bool exist = false;

	for(vector<frag>::iterator i = _frag.begin() ; i != _frag.end() ; i++)
	{
		if(newFrag.theSame(*i))
			exist = true;
	}

	return exist;
}

// get submatrix
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

// get sub point sequence
vector<Point> comp::subPointSeq(vector<Point> inputSeq, int startIndex, int matchL)
{
	vector<Point> result;

	for(int i = 0 ; i < matchL ; i++)
	{
		result.push_back(inputSeq[(startIndex+i)%inputSeq.size()]);
	}
	return result;
}

