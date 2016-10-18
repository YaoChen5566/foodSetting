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

Mat desToGrayImg(Mat input);
void compareDes(Mat input1, Mat input2, vector<Point> Seq1, vector<Point> Seq2);
Mat subMatrix(Mat input, int row, int col, int range);
vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);


int start1;
Mat des1;
vector<Point> pointSeq1;
int start2;
Mat des2;
vector<Point> pointSeq2;
int range;
float currentScore = 10000000;

int main()
{
	clock_t start = clock();
	string tmp = "foodImg/134.png";
	string tmp2 = "foodImg/142.png";
	descri descri1(tmp);
	Mat inputDes1 = descri1.resultDescri;
	descri descri2(tmp2);
	Mat inputDes2 = descri2.resultDescri;
	//cout <<"size: "<<descri1.sampleResult.size()<<endl;

	compareDes(inputDes1,inputDes2, descri1.sampleResult,descri2.sampleResult);
	//cout << "size: "<< descri1.sampleResult.size()<<endl;
	clock_t finish = clock();

	cout << "time: " << finish-start<<endl;

	Mat input1 = imread(tmp,-1);
	Mat input2 = imread(tmp2,-1);

	Mat input1_draw = input1.clone();
	Mat input2_draw = input2.clone();

	Mat warpingResult = input1.clone();

	//resize(input1, input1, Size(input1.cols*2, input1.rows*2) );
	//resize(input2, input2, Size(input2.cols*2, input2.rows*2) );

	for(int i = 0 ; i < range ; i++)
	{
		circle(input1_draw, pointSeq1[(start1+i)%pointSeq1.size()],1,Scalar(0,0,255,255),2);
		circle(input2_draw, pointSeq2[(start2+i)%pointSeq2.size()],1,Scalar(0,0,255,255),2);
		
	}
	//Point pt1 =  Point(10, 8);
	Mat des1RGB;
	Mat des2RGB;
	cvtColor(des1, des1RGB, CV_GRAY2RGB);
	cvtColor(des2, des2RGB, CV_GRAY2RGB);
	rectangle(des1RGB,Point(start1,start1) ,Point((start1+range)%pointSeq1.size(),(start1+range)%pointSeq1.size()), Scalar(0,0,255),1);
	rectangle(des2RGB,Point(start2,start2) ,Point((start2+range)%pointSeq2.size(),(start2+range)%pointSeq2.size()), Scalar(0,0,255),1);
	imwrite("result1.png", input1_draw);
	imwrite("result2.png", input2_draw);
	imwrite("des1.png", des1RGB);
	imwrite("des2.png", des2RGB);

	cout << "@start1: "<< start1<<" @start2: "<<start2<<" @range: "<<range<<endl;

	vector<Point> matchSeq1 = subPointSeq(pointSeq1, start1, range);
	vector<Point> matchSeq2 = subPointSeq(pointSeq2, start2, range);

	for(int i = 0 ; i < matchSeq1.size() ; i++)
	{
		cout <<"1: "<<matchSeq1[i]<<" 2: "<<matchSeq2[i]<<endl;
	}


	//cout << "size: "<<matchSeq1.size() << endl;
	Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, true); //(src, dst)
	cout <<"type: "<<warpingResult.type()<<endl;

	Mat vectorXY = Mat::ones(3, 1, CV_64FC1);
	Mat resultXY(2, 1, CV_32FC1);
	for(int i = 0 ; i < input2.rows ; i++)
	{
		for(int j = 0 ; j < input2.cols ; j++)
		{
			Vec4b dstBGRA;
			Vec4b & bgra = input2.at<Vec4b>(i, j);
			if(bgra[3] != 0) // not transparent
			{
				vectorXY.at<double>(0,0) = j;
				vectorXY.at<double>(1,0) = i;
				vectorXY.at<double>(2,0) = 1;
				
				//resultXY.at<float>(0, 0) = warp_mat.at<float>(0, 0)*vectorXY.at<float>(0,0) + warp_mat.at<float>(0, 1)*vectorXY.at<float>(1,0) + warp_mat.at<float>(0, 2)*vectorXY.at<float>(2,0);

				//resultXY.at<float>(1, 0) = warp_mat.at<float>(1, 0)*vectorXY.at<float>(0,0) + warp_mat.at<float>(1, 1)*vectorXY.at<float>(1,0) + warp_mat.at<float>(1, 2)*vectorXY.at<float>(2,0);

//				multiply(warp_mat, vectorXY, resultXY );
				Mat resultXY = warp_mat*vectorXY;
				dstBGRA[0] = bgra[0];
				dstBGRA[1] = bgra[1];
				dstBGRA[2] = bgra[2];
				dstBGRA[3] = bgra[3];
			
				if(int(resultXY.at<double>(0, 0)) >=0 && int(resultXY.at<double>(1, 0)) >= 0 && int(resultXY.at<double>(0, 0)) < warpingResult.rows && int(resultXY.at<double>(1, 0)) < warpingResult.cols)
				{
					//cout <<""
					warpingResult.at<Vec4b>((resultXY.at<double>(1, 0)),(resultXY.at<double>(0, 0))) = dstBGRA;
				}
			}

		}
	}

	imwrite("warping.png", warpingResult);
	//waitKey();
	system("Pause");
}

//get subPointSeq
vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range)
{
	vector<Point> result;

	for(int i = 0 ; i < range ; i++)
	{
		result.push_back(inputSeq[(startIndex+i)%inputSeq.size()]);
	}
	return result;
}

Mat desToGrayImg(Mat input)
{
	Mat tmp = Mat::zeros( input.rows, input.cols ,CV_32FC1);

	for(int i = 0 ; i < input.rows ; i++)
	{
		for(int j = 0 ; j < input.cols ; j++)
		{
			tmp.at<float>(i,j) = (input.at<float>(i,j)*255/180);
		}
	}
	return tmp;
}

Mat subMatrix(Mat input, int row, int col, int range)
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

// compare two descriptor
void compareDes(Mat input1, Mat input2, vector<Point> Seq1, vector<Point> Seq2)
{
	int r = 15; // square size
	int lefttopPoint1 = 0;
	int lefttopPoint2 = 0;
	//float currentScore = 10000000;
	
	double getScore = 0;

	for(int i = 0 ; i < input1.cols ; i++)
	{
		for(int j = 0 ; j < input2.cols ; j++)
		{
			Mat tmp1 = subMatrix(input1, i, i, r);
			Mat tmp2 = subMatrix(input2, j, j, r);

			Mat minus;
			absdiff(tmp1, tmp2, minus);
			//Scalar s = cv::sum(minus);
			
			getScore = cv::sum(minus)[0];
			/*for(int m = 0 ; m < r ; m++)
			{
				for(int n = 0 ; n < r ; n++)
				{
					getScore += pow((tmp1.at<float>(m, n) - tmp2.at<float>(m, n)), 2);
				}
			}*/
				
			getScore /= pow(r,2);

			if(getScore < currentScore)
			{
				 
				currentScore = getScore;
				start1 = i;
				des1 = input1;
				pointSeq1 = Seq1;
				pointSeq2 = Seq2;
				des2 = input2;
				start2 = j;
				range = r;
			}
		}
	}
	cout << "score: "<<currentScore<<endl;
}
