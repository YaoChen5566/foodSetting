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
	string tmp = "foodImg/148.png";
	string tmp2 = "foodImg/147.png";
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
		circle(input1_draw, pointSeq1[(start1+i)%pointSeq1.size()],1,Scalar(0,0,255,255),1);
		circle(input2_draw, pointSeq2[(start2+i)%pointSeq2.size()],1,Scalar(0,0,255,255),1);
		
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


	/*Mat test1 = Mat::ones(2,3,CV_32FC1);
	Mat test2 = Mat::ones(3,1,CV_32FC1);

	Mat test3 = test1*test2;*/

/*
	//src three points
	Point2f srcTri[3];
	//dst three points
	Point2f dstTri[3];

	// get affine matrix

	for(int i = 0 ; i < 3 ; i++)
	{
		cout << "1: "<< (start1+(range/2)*i)%pointSeq1.size() << " 2: "<< (start2+(range/2)*i)%pointSeq2.size() <<endl;
		srcTri[i] = pointSeq1[(start1+(range/2)*i)%pointSeq1.size()];
		dstTri[i] = pointSeq2[(start2+(range/2)*i)%pointSeq2.size()];
	}
	// warping matrix
	Mat warp_mat = Mat::ones( 2, 3, CV_64FC1 );
	warp_mat = getAffineTransform( srcTri, dstTri );
*/

	vector<Point> matchSeq1 = subPointSeq(pointSeq1, start1, range);
	vector<Point> matchSeq2 = subPointSeq(pointSeq2, start2, range);

	for(int i = 0 ; i < matchSeq1.size() ; i++)
	{
		cout <<"1: "<<matchSeq1[i]<<" 2: "<<matchSeq2[i]<<endl;
	}


	//cout << "size: "<<matchSeq1.size() << endl;
	Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, true); //(src, dst)


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
	
	float getScore = 0;

	for(int i = 0 ; i < input1.cols ; i++)
	{
		for(int j = 0 ; j < input2.cols ; j++)
		{
			Mat tmp1 = subMatrix(input1, i, i, r);
			Mat tmp2 = subMatrix(input2, j, j, r);

			for(int m = 0 ; m < r ; m++)
			{
				for(int n = 0 ; n < r ; n++)
				{
					getScore += pow((tmp1.at<float>(m, n) - tmp2.at<float>(m, n)), 2);
				}
			}
				
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

/*
Mat imgToDes(Mat input);
Mat alphaBinary(Mat input);
int maxContour(vector<vector<Point>> contours);
float contourLength(vector<Point> singleContour);
vector<Point> getSamplePoints(vector<Point> singleContour);
Mat descriptor(vector<Point> samplePoints);
Mat desToGrayImg(Mat input);
float angle(Point i, Point j, Point jMinusDelta);


int main()
{
	Mat input;
    input = imread("001.png", -1); 

	Mat des = imgToDes(input);
	Mat grayDes = desToGrayImg(des);
	imwrite("des.png", grayDes);

	
}

// inputImg to des
Mat imgToDes(Mat input)
{
	// resize input image
	Mat inputScaleTwo;
	resize(input, inputScaleTwo, Size(input.cols*2, input.rows*2) );

	//to binary image with alpha value
	Mat alpha = alphaBinary(inputScaleTwo);

	//color to Gray
	Mat inputGray;
	cvtColor(alpha, inputGray, CV_RGB2GRAY);

	//edge detection
	Mat inputCanny;
	Canny(inputGray, inputCanny, 50, 150, 3);
	
	// find contour
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(inputCanny, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
	Mat drawing = Mat::zeros( inputCanny.size(),CV_8UC1);

	//get sample points: points in vector
	vector<Point>tmpGG = getSamplePoints(contours[maxContour(contours)]);

	//get descriptor
	Mat shapeDes = descriptor(tmpGG);

	//descriptor to grayImg
	Mat grayDes = desToGrayImg(shapeDes);

	return shapeDes;
}

// alpha to binary
Mat alphaBinary(Mat input)
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
int maxContour(vector<vector<Point>> contours)
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
float contourLength(vector<Point> singleContour)
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
vector<Point> getSamplePoints(vector<Point> singleContour)
{
	int pointCount = 50;
	int totalPoints = singleContour.size();

	vector<int> pointIndex;
	vector<Point> samplePoints;
	int count = singleContour.size();
	int tmp = 0;

	//float step = totalPoints/pointCount;

	//cout <<"step: " <<step<<endl;
	srand(time(NULL));
	for(int i = 0 ; i < pointCount ; i++)
	{

		tmp = 0 + i*totalPoints/pointCount;
		//cout << tmp << endl;
		pointIndex.push_back(tmp);
	}

	sort(pointIndex.begin(),pointIndex.end());

	for(int i = 0 ; i < pointIndex.size() ; i++)
	{
		
		samplePoints.push_back(singleContour[pointIndex[i]]);
	}

	return samplePoints;
}

// get descriptor
Mat descriptor(vector<Point> samplePoints)
{
	Point pi;
	Point pj;
	Point pjMinusDelta;
	
	int delta = 3;
	float tmp = 0;
	Mat shapeDes = Mat::zeros(samplePoints.size(),samplePoints.size(),CV_32FC1); 
	for(int i = 0 ; i < samplePoints.size() ; i++)
	{
		for(int j = 0 ; j < samplePoints.size() ; j++)
		{
			pi = samplePoints[i];
			pj = samplePoints[j];
			
			
			if(abs(i-j) < delta)
			{
				shapeDes.at<float>(i,i) = 0;
			}
			else 
			{
				if(i > j)
					pjMinusDelta = samplePoints[(j+delta+samplePoints.size())%samplePoints.size()];
				else
					pjMinusDelta = samplePoints[(j-delta+samplePoints.size())%samplePoints.size()];

				tmp = angle(pi, pj, pjMinusDelta);
				//cout <<"i: "<<i<<", j:" <<j<<", angle: "<<255-(tmp*255/180) <<endl;
				shapeDes.at<float>(i,j) = (tmp*255/180);
				//shapeDes.at<float>(j,i) = 255-(tmp*255/180);
			}
		}
	}
	return shapeDes;
}

// normalize descriptor into 0~255
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

// get angle
float angle(Point i, Point j, Point jMinusDelta)
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
		angle = acos(cosAngle) * 180.0 / PI;
	}
	
	return angle;
}

*/
