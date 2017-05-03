#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>

#include "compare.h"
#include "fragment.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

ofstream myfile ("test.csv");

bool compareWithLength(map<string, int> input1, map<string, int> input2)
{
	int i = input1["l"];
	int j = input2["l"];
	return(i<j);
}

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

comp::comp(Mat descri1, vector<Mat> descri2Seq, vector<Point> pointSeq1, vector<Point> pointSeq2, int contourIndex, int foodIndex)
{

	_pointSeq1 = pointSeq1;
	_pointSeq2 = pointSeq2;
	_fIndex = foodIndex;
	_cIndex = contourIndex;
	_rDesSize = pointSeq1.size();
	_qDesSize = pointSeq2.size();
	setInitial();
	for(int i = 0 ; i < descri2Seq.size() ; i++)
		compareDesN(descri1, descri2Seq[i], i);

	localMaxOfRQMap();

	Mat mapRQ = _mapRQ.clone();
	//Mat mapRQ = normalizeRQ();

	imwrite("RQmap.png", mapRQ);
	//clearFrag();
	//disFrag();
	//localMin();
}

//set initial
void comp::setInitial()
{
	_thresholdScore = 500.0;
	_startIndex1 = 0;
	_startIndex2 = 0;
	_range = 0;
	_mapRQ = Mat::zeros(Size(_qDesSize, _rDesSize), CV_32S); //Size(q, r): x is q, y is r
	_mapScore = Mat::zeros(Size(_qDesSize, _rDesSize), CV_32S); //Size(q, r): x is q, y is r
	_warpMatrixMap.resize(_qDesSize, vector<Mat>(_rDesSize)); //Size(q, r): x id q, y is r
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
				//cout << getScore<<endl;
				if(getScore < _score)
				{
					//cout <<"i: "<<i<<" ,j: "<<j<<" ,r:"<<r<<endl;
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
	Mat smallerMat;
	Mat sub;

	if(input1.rows >= input2.rows)
	{
		smallerMat = input1(Rect(0, 0, input2.cols, input2.rows));
		sub = smallerMat - input2;
	}
	else
	{
		smallerMat = input2(Rect(0, 0, input1.cols, input1.rows));
		sub = input1 - smallerMat;
	}

	Mat integral1; // sum
	Mat integral2; // square sum

	int rLim = 0.4*sub.cols; // square size
	int lefttopPoint1 = 0;
	int lefttopPoint2 = 0;
	double tmpSum = 0;
	double getScore = 0;
	integral(sub, integral1, integral2);
	


	for(int i = 0 ; i < sub.cols ; i++)
	{
		for(int r = sub.cols ; r > rLim ; r--)
		{
			_range = r;
			_startIndex1 = i;
			_startIndex2 = (i+index)%input2.cols;

			//cout <<"startIndex1: "<<_startIndex1<<", startIndex2: "<<_startIndex2<<endl;

			if( (i+r) <= integral2.cols)
			{
				tmpSum = integral2.at<double>(i, i) + integral2.at<double>(i+r, i+r)-integral2.at<double>(i, i+r)-integral2.at<double>(i+r, i);
			
			}
			//else
			//{
			//	tmpSum += integral2.at<double>(i+r-input1.cols+1, i+r-input1.cols+1); //[y, y]
			//	tmpSum += integral2.at<double>(input1.cols-1, input1.cols-1)+integral2.at<double>(i, i)-integral2.at<double>(i, input1.cols-1)-integral2.at<double>(input1.cols-1, i); //[x, x]
			//	tmpSum += integral2.at<double>(i+r-input1.cols+1, input1.cols-1)-integral2.at<double>(i+r-input1.cols+1, i); //[y, cols]
			//	tmpSum += integral2.at<double>(input1.cols-1, i+r-input1.cols+1)-integral2.at<double>(i, i+r-input1.cols+1); //[cols, y]
			//}

			getScore = tmpSum/pow(r,2);
			

			if(getScore < _thresholdScore)
			{
				_score = getScore;

				// calculate the warping matrix
				vector<Point> matchSeqR = subPointSeq(_pointSeq1, _startIndex1, _range);
				vector<Point> matchSeqQ = subPointSeq(_pointSeq2, _startIndex2, _range);

				Mat warpMat = estimateRigidTransform(matchSeqQ, matchSeqR, false); // (src/query, dst/reference)

				// use scale and whether generate the warping matrix to judge the fragment probablity
				if(warpMat.size() != cv::Size(0, 0))
				{
					//cout << getScore<<endl;
					vector<Point> newPointSeq;
					
					for(int p = 0 ; p < _pointSeq2.size() ; p++)
					{
						double newX = warpMat.at<double>(0, 0)*_pointSeq2[p].x + warpMat.at<double>(0, 1)*_pointSeq2[p].y + warpMat.at<double>(0, 2);
						double newY = warpMat.at<double>(1, 0)*_pointSeq2[p].x + warpMat.at<double>(1, 1)*_pointSeq2[p].y + warpMat.at<double>(1, 2);
						newPointSeq.push_back(Point((int) newX, (int) newY));
					}

					double scale = pow(warpMat.at<double>(0, 0), 2) + pow(warpMat.at<double>(1, 0), 2);
					// add to RQmap
					if ( scale < 1.5 && scale > 0.5 &&  _mapRQ.at<int>(_startIndex1, _startIndex2)==0 && imageOverlap(newPointSeq))
					{
						_mapRQ.at<int>(_startIndex1, _startIndex2) = _range; 
						_mapScore.at<double>(_startIndex1, _startIndex2) = _score;
						_warpMatrixMap[_startIndex1][_startIndex2] = warpMat;
						break;
					}
						//if(_range >= _mapRQ.at<int>(_startIndex1, _startIndex2))
				}
			}
		}
	}
}

//single and a n seq 2
void comp::compareDesN2(Mat input1, Mat input2, int index)
{
	Mat sub = input1-input2;
	Mat integral1; // sum
	Mat integral2; // square sum

	int rLim = 0.4*input1.cols; // square size
	int lefttopPoint1 = 0;
	int lefttopPoint2 = 0;
	double tmpSum = 0;
	double getScore = 0;
	integral(sub, integral1, integral2);

	for(int r = input1.cols ; r > rLim ; r--)
	{
		//cout << r<<endl;
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

			_range = r;
			_score = getScore;
			_startIndex1 = i;
			_startIndex2 = i+index;		

			map<string, int> fragment;

			fragment["r"] = _startIndex1;
			fragment["q"] = _startIndex2;
			fragment["l"] = _range;
			fragment["fIndex"] = _fIndex;
			fragment["cIndex"] = _cIndex;
			fragment["score"] = getScore;

			//_frag.push_back(fragment);
			_totalFrag.push_back(fragment);

		}
	}
}

Mat comp::normalizeRQ()
{
	Mat tmp = Mat::zeros(_mapRQ.size(), CV_8UC1);
	double min, max;
	minMaxLoc(_mapRQ, &min, &max);

	for(int i = 0 ; i < _mapRQ.cols ; i++)
	{
		for(int j = 0 ; j < _mapRQ.rows ; j++)
		{
			tmp.at<char>(j, i) = _mapRQ.at<int>(j, i)*255/(max-min);
		}
	}

	return tmp;
}

// find the local maximum of RQ map
void comp::localMaxOfRQMap()
{
	// cut the RQ map into four part
	for(int i = 0 ; i < 2 ; i++) // q is x
	{
		for(int j = 0 ; j < 2 ; j++) // r is y
		{
			Rect roi_rect = Rect(0 + (i * _mapRQ.cols/2), 0 + (j * _mapRQ.rows/2), _mapRQ.cols/2, _mapRQ.rows/2);
			Mat roi = _mapRQ(roi_rect);
			double minVal;
			double maxVal;
			Point minLoc;
			Point maxLoc;

			minMaxLoc(roi, &minVal, &maxVal, &minLoc, &maxLoc);

			if(maxVal != 0)
			{
				frag fragMax;

				int tmpR = (int)maxLoc.y + (j * _mapRQ.rows/2);
				int tmpQ = (int)maxLoc.x + (i * _mapRQ.cols/2);

				fragMax.setInfo(tmpR, tmpQ, maxVal, _fIndex, _cIndex, _mapScore.at<double>(tmpR, tmpQ), _warpMatrixMap[tmpR][tmpQ]);
				_frag2.push_back(fragMax);

				map<string, int> fragment;
				
				fragment["r"] = (int)maxLoc.y + (j * _mapRQ.rows/2);
				fragment["q"] = (int)maxLoc.x + (i * _mapRQ.cols/2);
				fragment["l"] = maxVal;
				fragment["fIndex"] = _fIndex;
				fragment["cIndex"] = _cIndex;
				_frag.push_back(fragment);
			}
		}
	}
}

//preserve the best fragment for each length
void comp::clearFrag()
{
	vector< map<string, int> >::iterator iter1;
	double minScore[70];

	//cout <<"totalFrag Size: "<< _totalFrag.size() <<endl;

	sort(_totalFrag.begin(), _totalFrag.end(), compareWithLength);

	for(int i = 0 ; i < 70 ; i++)
	{
		minScore[i] = -1;
	}

	for(iter1 = _totalFrag.begin() ; iter1 != _totalFrag.end() ; iter1++)
	{
	
		if(minScore[(*iter1)["l"]-1] == -1)
		{
			minScore[(*iter1)["l"]-1] = (*iter1)["score"];
			_clearResult.push_back((*iter1));
		}
		else
		{
			if(minScore[(*iter1)["l"]-1] > (*iter1)["score"])
			{
				minScore[(*iter1)["l"]-1] = (*iter1)["score"];
				_clearResult.pop_back();
				_clearResult.push_back((*iter1));
			}
		}
	}

	vector< map<string, int> >::iterator iter2;
	
	//cout << "size: "<<_clearResult.size()<<endl;

	for(iter2 = _clearResult.begin() ; iter2 != _clearResult.end() ; iter2++)
	{
		//cout <<"l: "<< (*iter2)["l"] << ", score: " << (*iter2)["score"]<<endl;
		myfile << (*iter2)["l"] <<","<< (*iter2)["score"] <<endl;
	}

}

//dij
void comp::disFrag()
{
	//Mat disF;
	vector< map<string, int> >::iterator iter1; //i
	vector< map<string, int> >::iterator iter2; //j

	vector<int> vecI;
	vector<int> vecJ;

	vector<int> unionResult;
	vector<int>::iterator iterU;
	vector<int> intersectResult;
	vector<int>::iterator iterI;

	vector<vector<double> > distanceIJ(_clearResult.size(), vector<double>(_clearResult.size(), 0));

	double tmp;

	for(int c = 0 ; c < distanceIJ.size() ; c++ )
	{
		for(int r = 0 ; r < distanceIJ[c].size() ; r++)
		{
			for(int i = 0 ; i < _clearResult[c]["l"] ; i++ )
			{
				vecI.push_back((_clearResult[c]["r"]+i)%70);
			}
			for(int j = 0 ; j < _clearResult[r]["l"] ; j++ )
			{
				vecJ.push_back((_clearResult[r]["r"]+j)%70);
			}

			set_union(vecI.begin(), vecI.end(), vecJ.begin(), vecJ.end() , back_inserter(unionResult));
			set_intersection(vecI.begin(), vecI.end(), vecJ.begin(), vecJ.end(), back_inserter(intersectResult));
			
			tmp = (double)intersectResult.size()/(double)unionResult.size();
			//cout << c <<" "<< r <<" "<<tmp<<endl;
			distanceIJ[c][r] = tmp;

			vecI.clear();
			vecJ.clear();
			unionResult.clear();
			intersectResult.clear();

		}
	}

	for(int i = 0 ; i < distanceIJ.size() ; i++)
	{
		double tmp = 0;
		for(int j = 0 ; j < distanceIJ[i].size() ; j++)
		{
			tmp += distanceIJ[i][j];
		}
		tmp = tmp/distanceIJ.size();
		//cout << tmp <<endl;
		_clearResult[i]["score"] = _clearResult[i]["score"]*tmp;
		//myfile << _clearResult[i]["l"] <<","<< _clearResult[i]["score"] <<endl;
		//cout << _clearResult[i]["score"]<<endl;
	}

}

//get local minimum into _frag
void comp::localMin()
{
	vector< map<string, int> >::iterator iter2;
	for (iter2 = _clearResult.begin() + 1; iter2 != _clearResult.end() - 1; iter2++) {
		//cout << "l: " << (*iter2)["l"] << ", score: " << (*iter2)["score"] << endl;

		_startIndex1 = (*iter2)["r"];
		_startIndex2 = (*iter2)["q"];
		_range = (*iter2)["l"];
		
		if ((*(iter2 - 1))["score"] > (*iter2)["score"] && (*(iter2 + 1))["score"] > (*iter2)["score"])
		{
			// calculate the warping matrix
			vector<Point> matchSeqR = subPointSeq(_pointSeq1, _startIndex1, _range);
			vector<Point> matchSeqQ = subPointSeq(_pointSeq2, _startIndex2, _range);
			Mat warpMat = estimateRigidTransform(matchSeqQ, matchSeqR, false); // (src/query, dst/reference)

			if (warpMat.size() != cv::Size(0, 0))
			{
				//cout << "size"<<endl;
				double scale = pow(warpMat.at<double>(0, 0), 2) + pow(warpMat.at<double>(1, 0), 2);

				if ( abs(scale-1.0) < 1.5 /*&& scale > 0.2*/)
				{
					// use scale and whether generate the warping matrix to judge the fragment probablity
					map<string, int> fragment;
					fragment["r"] = (*iter2)["r"];
					fragment["q"] = (*iter2)["q"];
					fragment["l"] = (*iter2)["l"];
					fragment["fIndex"] = (*iter2)["fIndex"];
					fragment["cIndex"] = (*iter2)["cIndex"];
					fragment["score"] = (*iter2)["score"];
					_frag.push_back(fragment);
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
vector<map<string, int> > comp::fragList()
{
	return _frag;
}

// return a list of fragment with fragment.h
vector<frag> comp::fragList2()
{
	return _frag2;
}

// check if the same frag in the vector
bool comp::fragExist(map<string, int> newFrag)
{
	bool exist = false;

	for(vector<map<string, int> >::iterator i = _frag.begin() ; i != _frag.end() ; i++)
	{
		if(fragSame(newFrag, *i))
			exist = true;
	}

	return exist;
}

//check two fragment the same
bool comp::fragSame(map<string, int> frag1, map<string, int> frag2)
{
	if(frag1["r"] == frag2["r"] && frag1["q"] == frag2["q"])
		return true;
	else if(frag1["r"]-frag2["r"] == frag1["q"]-frag2["q"])
		return true;
	else
		return false;
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

	//result.push_back(inputSeq[startIndex%inputSeq.size()]);
	//result.push_back(inputSeq[(startIndex+matchL/2)%inputSeq.size()]);
	//result.push_back(inputSeq[(startIndex+matchL-1)%inputSeq.size()]);

	for(int i = 0 ; i < matchL ; i++)
	{
		result.push_back(inputSeq[(startIndex+i)%inputSeq.size()]);
	}
	return result;
}

/*
void comp::localMaxOfRQMap() {

	clock_t t1, t2;
	t1 = clock();
	get_Min_Max(_mapRQ, 14);
	t2 = clock();
	//cout << "get_min_max= " << (t2 - t1) << endl;

	for (int i = 0; i < _maxVec.size(); i++) {
		_startIndex1 = _maxVec[i].x;
		_startIndex2 = _maxVec[i].y;

		if (_mapRQ.at<int>(_startIndex1, _startIndex2) > 0) {
			//cout << "localMaxOfRQMap= " << _startIndex1 << ", " << _startIndex2 << endl;

			frag fragMax;
			fragMax.setInfo(_startIndex1, _startIndex2, _mapRQ.at<int>(_startIndex1, _startIndex2), _fIndex, _cIndex, _mapScore.at<int>(_startIndex1, _startIndex2), _warpMatrixMap[_startIndex1][_startIndex2]);
			_frag2.push_back(fragMax);
		}
	}
	//cout << "_frag.size= " << _frag.size();
}

// get min and max
void comp::get_Min_Max(Mat &rqmap, int windowSize) {
	int maximum, minimum;
	Point maxP, minP;
	_maxVec.clear();
	_minVec.clear();

	for (int i = 0; i < rqmap.cols; i += windowSize) {
		for (int j = 0; j < rqmap.rows; j += windowSize) {

			maximum = -1;
			minimum = 100;
			for (int m = 0; m < windowSize; m++) {
				for (int n = 0; n < windowSize; n++) {
					if (rqmap.at<int>(i + m, j + n) > maximum) {
						maximum = rqmap.at<int>(i + m, j + n);
						maxP = Point(i + m, j + n);
					}
					if (rqmap.at<int>(i + m, j + n) < minimum) {
						minimum = rqmap.at<int>(i + m, j + n);
						minP = Point(i + m, j + n);
					}
				}
			}
			_maxVec.push_back(maxP);
			_minVec.push_back(minP);
		}
	}
}
*/

bool comp::imageOverlap(vector<Point> newPointSeq)
{
	Size imgSize = Size(250, 250);

	

	/// Calculate the distances to the contour
	Mat contour_dist(imgSize, CV_32FC1);
	Mat food_dist(imgSize, CV_32FC1);
	Mat drawing = Mat::zeros(imgSize, CV_8UC1);
	Mat drawing2 = Mat::zeros(imgSize, CV_8UC1);
	Mat drawing3 = Mat::zeros(imgSize, CV_8UC1);
	int contourArea = 0, foodArea = 0, overlapArea = 0;


	for (int i = 0; i < imgSize.width; i++) {
		for (int j = 0; j < imgSize.height; j++) {
			contour_dist.at<float>(j, i) = pointPolygonTest(_pointSeq1, Point2f(i, j), true);
			food_dist.at<float>(j, i) = pointPolygonTest(newPointSeq, Point2f(i, j), true);

			//calculate contour area
			if (contour_dist.at<float>(j, i) > 0) {
				contourArea++;
				if (food_dist.at<float>(j, i) > 0) {
					overlapArea++;
					drawing.at<uchar>(j, i) = 255;
				}
				drawing2.at<uchar>(j, i) = 255;
			}
			if (food_dist.at<float>(j, i) > 0) {
				foodArea++;
				drawing3.at<uchar>(j, i) = 255;
			}
		}
	}
	imwrite("test/test_1.png", drawing);
	imwrite("test/test_2.png", drawing2);
	imwrite("test/test_3.png", drawing3);

	double ratio1 = (double)overlapArea / (double)contourArea;
	double ratio2 = (double)overlapArea / (double)foodArea;
	//cout << "ratio1= " << ratio1 << ", ratio2= " << ratio2 << endl;
	//cout << endl;

	if (/*ratio1 >= 0.5 &&*/ ratio2 >= 0.5)
	{
		cout << "ratio1= " << ratio1 << ", ratio2= " << ratio2 << endl;
		return true;
	}
	else
	{
		return false;
	}
}