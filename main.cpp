#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <dirent.h>
#include <errno.h>

#include "descri.h"
#include "compare.h"

# define PI 3.1415926

using namespace std;
using namespace cv;


vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);
int getdir(string dir, vector<string> &files);


int main()
{


	//Mat userDraw = imread("inputImg/man.jpg");
	//Mat userDrawGray;
	//cvtColor(userDraw, userDrawGray, CV_RGB2GRAY);
	//Mat userDrawCanny;
	//Canny(userDrawGray, userDrawCanny, 50, 150, 3);
	//vector<vector<Point>> userDrawContours;
	//vector<Vec4i> hierarchy;
	//findContours(userDrawCanny, userDrawContours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	//

	//string dir = string("foodImg/");
	//vector<string> files = vector<string>();
	//getdir(dir, files);
 // 
	//Mat drawing = Mat::zeros( userDraw.size(), CV_8UC3 );
	//drawContours( drawing, userDrawContours, 0 , Scalar(0,0,255), 2, 8, hierarchy, 0, Point() );
	//for(int i = 0 ; i < 1/*userDrawContours.size()*/ ; i++)
	//{
	//	descri descriUser(userDrawContours[i]);
	//	Mat userDrawDes = descriUser.resultDescri;
	//	
	//	for(int j = 2 ; j < files.size() ; j++)
	//	{
	//		string foodImg = dir + files[j];
	//		Mat food = imread(foodImg, -1);

	//		descri desFood(foodImg);
	//		Mat foodDes = desFood.resultDescri;
	//		comp compDes(userDrawDes,foodDes);

	//		vector<Point> matchSeq1 = subPointSeq(descriUser.sampleResult, compDes.startIndex1, compDes.range);
	//		vector<Point> matchSeq2 = subPointSeq(desFood.sampleResult, compDes.startIndex2, compDes.range);

	//		Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, false); //(src, dst)
	//		//cout <<"type: "<<warpingResult.type()<<endl;
	//		//cout << warp_mat.size()<<endl;
	//		if(warp_mat.size() != cv::Size(0,0))
	//		{
	//			cout << "file: "<< files[j]<<endl;
	//			cout << "score: "<<compDes.score<<endl;;
	//			cout << "scale: "<< pow(warp_mat.at<double>(0,0), 2) + pow(warp_mat.at<double>(1,0), 2)  <<endl;
	//			warpAffine(food, userDraw, warp_mat, food.size());
	//		}

	//	}
	//}



	clock_t start = clock(); // compare start
	string tmp = "foodImg/085.png";
	string tmp2 = "foodImg/084.png";
	Mat input1 = imread(tmp, -1);
	Mat input2 = imread(tmp2, -1);

	descri descri1(tmp);
	Mat inputDes1 = descri1.resultDescri;
	descri descri2(tmp2);
	Mat inputDes2 = descri2.resultDescri;

	comp compDes(inputDes1,inputDes2);

	clock_t finish = clock(); // compare finish

	cout << "time: " << finish-start<<endl;
	
	Mat input1_draw = input1.clone();
	Mat input2_draw = input2.clone();

	Mat warpingResult = input1.clone();

	vector<Point> pointSeq1 = descri1.sampleResult; 
	vector<Point> pointSeq2 = descri2.sampleResult;

	for(int i = 0 ; i < compDes.range ; i++)
	{
		circle(input1_draw, pointSeq1[(compDes.startIndex1+i)%pointSeq1.size()],1,Scalar(0,0,255,255),2);
		circle(input2_draw, pointSeq2[(compDes.startIndex2+i)%pointSeq2.size()],1,Scalar(0,0,255,255),2);	
	}
	Mat des1RGB;
	Mat des2RGB;
	cvtColor(inputDes1, des1RGB, CV_GRAY2RGB);
	cvtColor(inputDes2, des2RGB, CV_GRAY2RGB);
	rectangle(des1RGB,Point(compDes.startIndex1,compDes.startIndex1) ,Point((compDes.startIndex1+compDes.range)%pointSeq1.size(),(compDes.startIndex1+compDes.range)%pointSeq1.size()), Scalar(0,0,255),1);
	rectangle(des2RGB,Point(compDes.startIndex2,compDes.startIndex2) ,Point((compDes.startIndex2+compDes.range)%pointSeq2.size(),(compDes.startIndex2+compDes.range)%pointSeq2.size()), Scalar(0,0,255),1);
	imwrite("result1.png", input1_draw);
	imwrite("result2.png", input2_draw);
	imwrite("des1.png", des1RGB);
	imwrite("des2.png", des2RGB);

	cout << "@start1: "<< compDes.startIndex1 <<" @start2: "<< compDes.startIndex2 <<" @range: "<< compDes.range <<endl;

	vector<Point> matchSeq1 = subPointSeq(pointSeq1, compDes.startIndex1, compDes.range);
	vector<Point> matchSeq2 = subPointSeq(pointSeq2, compDes.startIndex2, compDes.range);

	for(int i = 0 ; i < matchSeq1.size() ; i++)
	{
		cout <<"1: "<<matchSeq1[i]<<" 2: "<<matchSeq2[i]<<endl;
	}


	Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, false); //(src, dst)
	cout <<"type: "<<warp_mat.size()<<endl;
	cout <<"scale: "<< pow(warp_mat.at<double>(0,0), 2) + pow(warp_mat.at<double>(1,0), 2)  <<endl;
	//Mat vectorXY = Mat::ones(3, 1, CV_64FC1);
	//Mat resultXY(2, 1, CV_32FC1);
	//for(int i = 0 ; i < input2.rows ; i++)
	//{
	//	for(int j = 0 ; j < input2.cols ; j++)
	//	{
	//		Vec4b dstBGRA;
	//		Vec4b & bgra = input2.at<Vec4b>(i, j);
	//		if(bgra[3] != 0) // not transparent
	//		{
	//			vectorXY.at<double>(0,0) = j;
	//			vectorXY.at<double>(1,0) = i;
	//			vectorXY.at<double>(2,0) = 1;

	//			Mat resultXY = warp_mat*vectorXY;
	//			dstBGRA[0] = bgra[0];
	//			dstBGRA[1] = bgra[1];
	//			dstBGRA[2] = bgra[2];
	//			dstBGRA[3] = bgra[3];
	//		
	//			if(int(resultXY.at<double>(0, 0)) >=0 && int(resultXY.at<double>(1, 0)) >= 0 && int(resultXY.at<double>(0, 0)) < warpingResult.rows && int(resultXY.at<double>(1, 0)) < warpingResult.cols)
	//				warpingResult.at<Vec4b>((resultXY.at<double>(1, 0)),(resultXY.at<double>(0, 0))) = dstBGRA;
	//		
	//		}

	//	}
	//}
	
	warpAffine(input2, warpingResult, warp_mat, warpingResult.size());
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

int getdir(string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL)
	{
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }
    while((dirp = readdir(dp)) != NULL)
	{
		files.push_back(string(dirp->d_name));
    }
    closedir(dp);
    return 0;
}