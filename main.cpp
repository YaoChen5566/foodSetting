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
#include <map>

#include "tree.hh"

#include "descri.h"
#include "compare.h"
//#include "fragment.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

//
//vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);
int getdir(string dir, vector<string> &files);


// comparison function object
bool compareContourSize ( vector<Point> contour1, vector<Point> contour2 ) {
	size_t i = contour1.size();
	size_t j = contour2.size();
    return ( i > j );
}

struct fragm 
{  
   map<string, int> Element;  
}; 

struct fragCanList
{
	vector<map<string, int>> Element;
};

struct foodFragList
{
	map<string, fragCanList> Element;
};

struct contourList
{
	map<int, foodFragList> Element;
};

int main()
{
	
	//fragm test;
	//test["FF"] = 1;

	//map<string, int> ttt;
	//ttt["FF"] = 1;

	Mat userDraw = imread("inputImg/rabbit.jpg");
	Mat userDraw2;
	resize(userDraw, userDraw2, Size(userDraw.cols*2, userDraw.rows*2));
	Mat userDrawGray;
	cvtColor(userDraw2, userDrawGray, CV_BGR2GRAY);
	Mat userDrawCanny;
	Canny(userDrawGray, userDrawCanny, 25, 75, 3);
	Mat userDrawCannyT;
	threshold(userDrawCanny,userDrawCannyT,120,255,CV_THRESH_BINARY);


	//cout << userDraw.type();
	Mat userDrawFloat;
	userDraw.convertTo(userDrawFloat, CV_32FC3);
	imwrite("float.png", userDrawFloat);
	//userDraw.convertTo(userDrawFloat, CV_32FC3);
	//Mat userDrawRe = userDraw.reshape((userDraw.rows * userDraw.cols, 1));



	//cout << "1";
	vector<Mat> channels;
	split(userDraw, channels);

	Mat B = channels[0];
	Mat G = channels[1];
	Mat R = channels[2];

	Mat cannyB, cannyG, cannyR;

	Canny(B, cannyB, 50, 150, 3);
	Canny(G, cannyG, 50, 150, 3);
	Canny(R, cannyR, 50, 150, 3);

	Mat cannyColor;

	bitwise_or(cannyB, cannyG, cannyColor);
	bitwise_or(cannyColor, cannyR, cannyColor);


	//for(int i = 0 ; i < userDraw.cols ; i++)
	//{
	//	for(int j = 0 ; j < userDraw.rows ; j++)
	//	{
	//		cout <<"B: "<<B.at<uchar>(j, i)<<" , G: "<<G.at<uchar>(j, i)<<" , R: "<<R.at<uchar>(j, i)<<endl;
	//		inputSpace[B.at<uchar>(j, i)][G.at<uchar>(j, i)][R.at<uchar>(j, i)] += 1;
	//	}
	//}

	vector<vector<Point>> userDrawContours;
	vector<Vec4i> hierarchy;

	findContours(cannyColor.clone(), userDrawContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	
	
	string dir = string("foodImg/");
	vector<string> files = vector<string>();
	getdir(dir, files);
  
	vector<vector<Point>> disjointContour;

	for(int i = 0 ; i < userDrawContours.size() ; i++)
	{
		if(userDrawContours[i].size()>50 && hierarchy[i][3] != -1)
			disjointContour.push_back(userDrawContours[i]);
	}

	sort(disjointContour.begin(), disjointContour.end(), compareContourSize);


	


	RNG rng(12345);
	Mat drawing = Mat::zeros( userDraw.size(), CV_8UC3 );
	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		cout << disjointContour[i].size()<<endl;;
		drawContours( drawing, disjointContour, i ,  Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) ), 2, 8);
	}

	imwrite("canny.png", cannyColor);
	imwrite("contour.png", drawing);
	

	contourList contourCandidate; // key: contour index, value: map within the information of file and fragment
	foodFragList fileFragment; //key: file, value: vector of fragment

	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		
		descri descriUser(userDrawContours[i]);
		Mat userDrawDes = descriUser.resultDescri();
		
		for(int j = 2 ; j < files.size() ; j++)
		{
			string foodImg = dir + files[j];
			Mat food = imread(foodImg, -1);

			descri desFood(foodImg);
			vector<Mat> foodDes = desFood.seqDescri();

			comp compDes(userDrawDes,foodDes, descriUser.sampleResult(), desFood.sampleResult());

			fragCanList compFragList;
			compFragList.Element = compDes.fragList();

			cout <<"file: "<<files[j]<<endl;
			fileFragment.Element[foodImg] = compFragList;
			if(compFragList.Element.size() > 0)
			{

				for(int k = 0 ; k < compFragList.Element.size() ; k++)
				{
					//cout <<"file: "<<files[j]<<endl;
					cout <<"reference index: "<<compFragList.Element[k]["r"]<<endl;
					cout <<"query index: "<<compFragList.Element[k]["q"]<<endl;
					cout <<"match length: "<<compFragList.Element[k]["l"]<<endl;

				}
			}
		}
		contourCandidate.Element[i] = fileFragment;	
	}


	/*
	// start warping
	map<int, map<string, vector<frag>>>::iterator iter1;
	map<string, vector<frag>>::iterator iter2;
	vector<frag>::iterator iter3;

	tree<string> tr;




	for(iter1 = contourCandidate.begin() ; iter1 != contourCandidate.end() ; iter1++)
	{
		for(iter2 = fileFragment.begin() ; iter2 != fileFragment.end() ; iter2++)
		{
			for(iter3 = fileFragment[iter2 -> first].begin() ; iter3 != fileFragment[iter2 -> first].end() ; iter3++)
			{
			}
		}
	}
	*/


	/*
	
	clock_t start = clock(); // compare start
	string tmp = "foodImg/085.png";
	string tmp2 = "foodImg/084.png";
	Mat input1 = imread(tmp, -1);
	Mat input2 = imread(tmp2, -1);

	descri descri1(tmp);
	Mat inputDes1 = descri1.resultDescri();
	descri descri2(tmp2);
	Mat inputDes2 = descri2.resultDescri();
	vector<Mat> inputDesSeq2 = descri2.seqDescri();

	cout << inputDesSeq2.size()<<endl;;

	//comp compDes(inputDes1,inputDes2);
	comp compDes(inputDes1, inputDesSeq2);

	clock_t finish = clock(); // compare finish

	cout << "time: " << finish-start<<endl;
	
	cout << "@start1: "<< compDes.startIndex1() <<" @start2: "<< compDes.startIndex2() <<" @range: "<< compDes.range() <<" @score: "<< compDes.score()<<endl;

	Mat input1_draw = input1.clone();
	Mat input2_draw = input2.clone();

	Mat warpingResult = input1.clone();

	vector<Point> pointSeq1 = descri1.sampleResult(); 
	vector<Point> pointSeq2 = descri2.sampleResult();

	for(int i = 0 ; i < compDes.range() ; i++)
	{
		circle(input1_draw, pointSeq1[(compDes.startIndex1()+i)%pointSeq1.size()],1,Scalar(0,0,255,255),2);
		circle(input2_draw, pointSeq2[(compDes.startIndex2()+i)%pointSeq2.size()],1,Scalar(0,0,255,255),2);	
	}

	Mat des1RGB;
	Mat des2RGB;
	cvtColor(inputDes1, des1RGB, CV_GRAY2RGB);
	cvtColor(inputDes2, des2RGB, CV_GRAY2RGB);
	
	imwrite("result1.png", input1_draw);
	imwrite("result2.png", input2_draw);

	vector<Point> matchSeq1 = subPointSeq(pointSeq1, compDes.startIndex1(), compDes.range());
	vector<Point> matchSeq2 = subPointSeq(pointSeq2, compDes.startIndex2(), compDes.range());

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
	*/

	//waitKey();
	system("Pause");
}

//get subPointSeq
//vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range)
//{
//	vector<Point> result;
//
//	for(int i = 0 ; i < range ; i++)
//	{
//		result.push_back(inputSeq[(startIndex+i)%inputSeq.size()]);
//	}
//	return result;
//}

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