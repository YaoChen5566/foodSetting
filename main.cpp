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
#include <fstream>
#include <map>

#include "tree.hh"

#include "descri.h"
#include "compare.h"

//#include "fragment.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

//all files(food image) in the dir
vector<string> files = vector<string>();

//single test
void singleTest(void);
//warp test
void warpTest(void);


//get dir files
int getdir(string dir, vector<string> &files);

//segment image with alpha value
Mat alphaBinary(Mat input);

//three channels Canny
Mat cannyThreeCh(Mat input);

// edge error
double edgeError(Mat draw, Mat food);

//color error
double colorError(Mat draw, Mat food);

//reference error
double refError(Mat draw, Mat food, int& nextIndex);

//subPointSeq
vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);

//add image with transparent background
Mat addTransparent(Mat &bg, Mat &fg);

// return the min error value index of fragment
int minValueInMap( vector<map<string, int> > input);

//return the vector of seqence of error value from small to large
vector<int> vecSeqIndex(vector<map<string, int> > input);

void preProcess(vector<vector<Mat> >* p_foodDesSeq, vector<vector<Point> >* p_sampleResult);

//read food descriptor and point
vector<Mat> vecmatread(const string& filename);
vector<Point> vecPointRead(const string& filename);

// comparison function object
bool compareContourSize ( vector<Point> contour1, vector<Point> contour2 ) {
	size_t i = contour1.size();
	size_t j = contour2.size();
    return ( i > j );
}



struct fragList
{
	vector<map<string, int> > Element;
};

struct cfMap
{
	map<int, fragList> Element;
};

int main()
{
	Mat userDraw = imread("inputImg/inin.png", -1);

	Mat cannyColor = cannyThreeCh(userDraw);

	vector<vector<Point> > userDrawContours;
	vector<Vec4i> hierarchy;

	findContours(cannyColor.clone(), userDrawContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	
	vector<vector<Point> > disjointContour;

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
		cout << disjointContour[i].size()<<endl;
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, disjointContour, i ,  color, 1, 8);
	}

	imwrite("canny.png", cannyColor);
	imwrite("contour.png", drawing);

	string dir = string("foodImg/");
	//vector<string> files = vector<string>();
	getdir(dir, files);

	// pre-process for contour descriptor and sample points
	cout <<"start pre-process"<<endl;
	
	vector<Mat> desOfDraw;
	vector<vector<Point> > samplepointsOfDraw;
	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		descri descriUser(disjointContour[i]);
		desOfDraw.push_back(descriUser.resultDescri());
		samplepointsOfDraw.push_back(descriUser.sampleResult());
	}

	//pre-Process for food descriptor and sample points
	vector<vector<Mat> > desOfFood;
	vector<vector<Point> > samplepointsOfFood;
	preProcess(&desOfFood, &samplepointsOfFood);


	cout << "pre-process done"<<endl;

	int fragNum = 0;

	cfMap foodCandidate;

	fragList pairSeq;
	

	for(int i = 0 ; i < desOfDraw.size() ; i++)
	{
		cout << disjointContour[i].size()<<endl;
		
		clock_t start = clock(); // compare start

		for(int j = 2 ; j < files.size() ; j++)
		{

			comp compDes(desOfDraw[i],desOfFood[j-2], samplepointsOfDraw[i], samplepointsOfFood[j-2], i, j-2);
			//comp compDes(desOfDraw[i], desFood.seqDescri(), samplepointsOfDraw[i], desFood.sampleResult(), i, j);

			fragList tmpPairSeq; 
			tmpPairSeq.Element = compDes.fragList();

			//cout <<"file: "<<files[j]<<endl;

			if(tmpPairSeq.Element.size() > 0)
			{
				fragNum += tmpPairSeq.Element.size();

				for(int k = 0 ; k < tmpPairSeq.Element.size() ; k++)
				{
					//cout <<"contour: "<<tmpPairSeq.Element[k]["cIndex"]<<endl;
					//cout <<"file: "<<files[tmpPairSeq.Element[k]["fIndex"]+2]<<endl;
					//cout <<"reference index: "<<tmpPairSeq.Element[k]["r"]<<endl;
					//cout <<"query index: "<<tmpPairSeq.Element[k]["q"]<<endl;
					//cout <<"match length: "<<tmpPairSeq.Element[k]["l"]<<endl;
					
					//warping and save the error value for each fragment
					vector<Point> matchSeq1 = subPointSeq(samplepointsOfDraw[i], tmpPairSeq.Element[k]["r"], tmpPairSeq.Element[k]["l"]);
					vector<Point> matchSeq2 = subPointSeq(samplepointsOfFood[j-2], tmpPairSeq.Element[k]["q"], tmpPairSeq.Element[k]["l"]);
					//vector<Point> matchSeq2 = subPointSeq(desFood.sampleResult(), tmpPairSeq.Element[k]["q"], tmpPairSeq.Element[k]["l"]);

					string foodImg = dir + files[j];
					Mat food = imread(foodImg, -1);

					Mat foodStack = Mat::zeros(userDraw.size(), CV_8UC4);
					Mat drawClone = Mat::zeros(userDraw.size(), CV_8UC4);
					vector< vector<Point> > tmpContour;
					tmpContour.push_back(samplepointsOfFood[j-2]);
					//tmpContour.push_back(desFood.sampleResult());

					Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, false); //(src, dst)

					//Mat drawClone = userDraw.clone();
					warpAffine(food, foodStack, warp_mat, foodStack.size());

					int tmpp;
					tmpPairSeq.Element[k]["eError"] = edgeError(userDraw, foodStack);
					tmpPairSeq.Element[k]["cError"] = colorError(userDraw, foodStack);
					tmpPairSeq.Element[k]["rError"] = refError(userDraw, foodStack, tmpp);
					tmpPairSeq.Element[k]["sError"] = tmpPairSeq.Element[k]["eError"] + tmpPairSeq.Element[k]["cError"] + tmpPairSeq.Element[k]["rError"];
					
				}
			}
			pairSeq.Element.insert(pairSeq.Element.end(), tmpPairSeq.Element.begin(), tmpPairSeq.Element.end());




		}
		foodCandidate.Element[i] = pairSeq;	
		pairSeq.Element.clear();
		
		clock_t finish = clock(); // compare finish

		cout << "time: " << finish-start<<endl;
		//cout <<i<<": " <<pairSeq.Element.size()<<endl;
	}
	
	for(int i = 0 ; i < disjointContour.size() ; i++)
		cout <<i<<"'s candidate" <<foodCandidate.Element[i].Element.size()<<endl;

	vector<int> contourSeq;
	vector<int> fragmentSeq;

	int nextIndex = 0;
	int nextFrag = 0;

	int preNextIndex = 0;
	int preNextFrag = 0;

	vector<vector<int> > contourMatchSeq;

	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		contourMatchSeq.push_back(vecSeqIndex(foodCandidate.Element[i].Element));
	}

	nextFrag = contourMatchSeq[nextIndex][0];

	contourSeq.push_back(nextIndex);
	fragmentSeq.push_back(nextFrag);

	//get contour and food subsequence
	vector<Point> matchSeqC = subPointSeq(samplepointsOfDraw[foodCandidate.Element[nextIndex].Element[nextFrag]["cIndex"]], foodCandidate.Element[nextIndex].Element[nextFrag]["r"], foodCandidate.Element[nextIndex].Element[nextFrag]["l"]);
	vector<Point> matchSeqF = subPointSeq(samplepointsOfFood[foodCandidate.Element[nextIndex].Element[nextFrag]["fIndex"]], foodCandidate.Element[nextIndex].Element[nextFrag]["q"], foodCandidate.Element[nextIndex].Element[nextFrag]["l"]);

	//get warping matrix
	Mat warpMat = estimateRigidTransform(matchSeqF, matchSeqC, false); //(src, dst)

	Mat resultStack = userDraw.clone();

	warpAffine(imread(dir+files[foodCandidate.Element[nextIndex].Element[nextFrag]["fIndex"]+2], -1), resultStack, warpMat, userDraw.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);

	Mat result = addTransparent(userDraw, resultStack);
	Mat resultStackClone = resultStack.clone();
	
	int fragPtr = 0;
	double totalErr, preErr;
	totalErr = edgeError(userDraw, resultStack) + colorError(userDraw, resultStack) + refError(userDraw, resultStack, nextIndex);
	cout <<"totalErr: " <<totalErr<<endl;
	cout <<"!!!"<<endl;
	preErr = totalErr;
	
	while(totalErr > 10)
	{
		if(preErr >= totalErr)
		{
			preErr = totalErr;
			cout <<"nextIndex: "<<nextIndex<<endl;
			nextFrag = contourMatchSeq[nextIndex][fragPtr];
			cout <<"nextFood: "<<nextFrag<<endl;
		}
		else
		{
			fragPtr++;
			if(fragPtr > contourMatchSeq[nextIndex].size())
				break;

			nextFrag = contourMatchSeq[nextIndex][fragPtr];
			nextIndex = contourSeq.back();
			contourSeq.pop_back();
			fragmentSeq.pop_back();
			resultStack = resultStackClone.clone();
		}

		contourSeq.push_back(nextIndex);
		fragmentSeq.push_back(nextFrag);

		vector<Point> matchSeqDraw = subPointSeq(samplepointsOfDraw[foodCandidate.Element[nextIndex].Element[nextFrag]["cIndex"]], foodCandidate.Element[nextIndex].Element[nextFrag]["r"], foodCandidate.Element[nextIndex].Element[nextFrag]["l"]);
		vector<Point> matchSeqFood = subPointSeq(samplepointsOfFood[foodCandidate.Element[nextIndex].Element[nextFrag]["fIndex"]], foodCandidate.Element[nextIndex].Element[nextFrag]["q"], foodCandidate.Element[nextIndex].Element[nextFrag]["l"]);

		Mat warpMat_2 = estimateRigidTransform(matchSeqFood, matchSeqDraw, false);
		Mat resultStack_2 = resultStack.clone();
		Mat warpFood = imread(dir+files[foodCandidate.Element[nextIndex].Element[nextFrag]["fIndex"]+2], -1);
		
		warpAffine(warpFood, resultStack_2, warpMat_2, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);

		resultStackClone = resultStack.clone();
		resultStack = addTransparent(resultStack, resultStack_2);

		totalErr = edgeError(userDraw, resultStack) + colorError(userDraw, resultStack) + refError(userDraw, resultStack, nextIndex);
		cout <<"totalErr: " <<totalErr<<"preErr: "<<preErr<<endl;
	}
	
	system("Pause");
}

//single test
void singleTest(void)
{
	clock_t start = clock(); // compare start
	string tmp = "foodImg/113.png";
	string tmp2 = "foodImg/031.png";
	Mat input1 = imread(tmp, -1);
	Mat input2 = imread(tmp2, -1);

	descri descri1(tmp);
	Mat inputDes1 = descri1.resultDescri();
	descri descri2(tmp2);
	Mat inputDes2 = descri2.resultDescri();
	vector<Mat> inputDesSeq2 = descri2.seqDescri();

	cout << inputDesSeq2.size()<<endl;;

	//comp compDes(inputDes1,inputDes2);
	comp compDes(inputDes1, inputDesSeq2, descri1.sampleResult(), descri2.sampleResult(), 0, 0);

	vector<map<string, int> > tmpppp = compDes.fragList();
	clock_t finish = clock(); // compare finish

	cout << "time: " << finish-start<<endl;
	//cout << "size: " << compDes.fragList().size()<<endl;
	cout << "@start1: "<< compDes.startIndex1() <<" @start2: "<< compDes.startIndex2() <<" @range: "<< compDes.range() <<endl;

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


	
	imwrite("result1.png", input1_draw);
	imwrite("result2.png", input2_draw);


	//vector<map<string, int>>::iterator iterV;
	//map<string, int>::iterator iterM;


	vector<Point> matchSeq1 = subPointSeq(pointSeq1, compDes.startIndex1(), compDes.range());
	vector<Point> matchSeq2 = subPointSeq(pointSeq2, compDes.startIndex2(), compDes.range());

	//for(int i = 0 ; i < matchSeq1.size() ; i++)
	//{
	//	cout <<"1: "<<matchSeq1[i]<<" 2: "<<matchSeq2[i]<<endl;
	//}


	Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, false); //(src, dst)


	cout <<"type: "<<warp_mat.size()<<endl;
	cout <<"scale: "<< pow(warp_mat.at<double>(0,0), 2) + pow(warp_mat.at<double>(1,0), 2)  <<endl;
	
	warpAffine(input2, warpingResult, warp_mat, warpingResult.size());
	imwrite("warping.png", warpingResult);
	

}

//warp test
void warpTest(void)
{
	Mat userDraw = imread("inputImg/inin.png");
	Mat food = imread("foodImg/008.png", -1);
	Mat userDraw2 = userDraw.clone();
	Point center = Point(food.cols/2, food.rows/2);
    double angle = 30.0;
    double scale = 0.8;

    Mat rot_mat = getRotationMatrix2D(center, angle, scale);

	warpAffine(food, userDraw2, rot_mat, userDraw.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	Mat result = addTransparent(userDraw, userDraw2);
	imwrite("result.png", result);
}

//add image with transparent background
Mat addTransparent(Mat &bg, Mat &fg)
{
	Mat result;
	bg.copyTo(result);
	for(int y = 0; y < bg.rows; ++y)
	{
		int fY = y;
		if(fY >= fg.rows) break;
		for(int x = 0; x < bg.cols; ++x)
		{
			int fX = x;
			if(fX >= fg.cols) break;
			double Fopacity =((double)fg.data[fY * fg.step + fX * fg.channels() + 3]) / 255; // opacity of the foreground pixel

			for(int c = 0; Fopacity > 0 && c < result.channels(); ++c) // combine the background and foreground pixel
			{
				unsigned char foregroundPx =fg.data[fY * fg.step + fX * fg.channels() + c];
				unsigned char backgroundPx =bg.data[y * bg.step + x * bg.channels() + c];
				result.data[y*result.step + result.channels()*x + c] =backgroundPx * (1.-Fopacity) + foregroundPx * Fopacity;

			}
		}
	}
	return result;
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

// canny edge detection for each channel
Mat cannyThreeCh(Mat input)
{
	vector<Mat> channels;
	split(input, channels);

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

	return cannyColor;
}

// get all files in the dir
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

//segment image with alpha value
Mat alphaBinary(Mat input)
{
	Mat alphaOrNot = Mat::zeros(input.size(),CV_32S);
	for(int i = 0 ; i < input.cols ; i++)
	{
		for(int j = 0 ; j < input.rows ; j++)
		{
			Vec4b & bgra = input.at<Vec4b>(j, i);
			if(bgra[3] != 0) // not transparency
				alphaOrNot.at<int>(j, i) = 1;
			else
				alphaOrNot.at<int>(j, i) = 0;
			
		}
	}
	return alphaOrNot;
}

//edge error
double edgeError(Mat draw, Mat food)
{
	Mat drawEdge = cannyThreeCh(draw);
	Mat foodEdge = cannyThreeCh(food);

	vector<vector<Point>> drawSeqContours;
	vector<Vec4i> hierarchyD;

	vector<vector<Point>> foodSeqContours;
	vector<Vec4i> hierarchyF;

	findContours(drawEdge.clone(), drawSeqContours, hierarchyD, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	findContours(foodEdge.clone(), foodSeqContours, hierarchyF, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );

	int foodPointNum = 0;

	for(int i = 0 ; i < foodSeqContours.size() ; i++)
	{
		foodPointNum += int(foodSeqContours[i].size());
	}

	Mat drawDrawContour = Mat::zeros(draw.size(), CV_32FC3);
	Mat foodDrawContour = Mat::zeros(food.size(), CV_32FC3);

	for(int i = 0 ; i < drawSeqContours.size() ; i++)
		drawContours( drawDrawContour, drawSeqContours, i, Scalar(255, 255, 255), 1, 8);
	
	//for(int i = 0 ; i < foodSeqContours.size() ; i++)
		//drawContours( foodDrawContour, foodSeqContours, i, Scalar(255, 255, 255), 1, 8);

	Mat drawConGray;
	//Mat foodConGray;
	
	cvtColor(drawDrawContour, drawConGray, CV_BGR2GRAY);
	//cvtColor(foodDrawContour, foodConGray, CV_BGR2GRAY);

	Mat drawBin = drawConGray > 128;
	//Mat foodBin = foodConGray > 128;

	Mat nonZeroDraw;
	//Mat nonZeroFood;
	findNonZero(drawBin, nonZeroDraw);
	//findNonZero(foodBin, nonZeroFood);
	
	vector<double> pointDist;

	double score = 0;

	for(int i = 0 ; i < foodSeqContours.size() ; i++)
	{
		for(int j = 0 ; j < foodSeqContours[i].size() ; j++)
		{
			Point locF = foodSeqContours[i][j];
			
			for(int k = 0 ; k < nonZeroDraw.rows ; k++)
			{
				Point locD = nonZeroDraw.at<Point>(k);
				pointDist.push_back(norm(locF-locD));
			}
			double tmp = *min_element(pointDist.begin(), pointDist.end());
			//cout << tmp << endl;
			score += tmp;
			pointDist.clear();
		}
	}

	return score/foodPointNum;
}

//color error
double colorError(Mat draw, Mat food)
{
	double score = 0.0;

	Mat drawAlphaBin = alphaBinary(draw);
	Mat foodAlphaBin = alphaBinary(food);

	int orValue;
	double tmp;
	for(int i = 0 ; i < drawAlphaBin.rows ; i++)
	{
		for(int j = 0 ; j < drawAlphaBin.cols ; j++)
		{
			orValue = drawAlphaBin.at<int>(i, j) | foodAlphaBin.at<int>(i, j);

			if(orValue == 1)
			{
				Vec4b pixDraw = draw.at<Vec4b>(i, j);
				Vec4b pixFood = food.at<Vec4b>(i, j);

				tmp =  sqrt(pow(pixDraw.val[0]-pixFood.val[0], 2) + pow(pixDraw.val[1]-pixFood.val[1], 2) + pow(pixDraw.val[2]-pixFood.val[2], 2));
				//cout << score << endl;

				score += tmp;

			}
		}
	}

	return score/(draw.rows*draw.cols);
}

//reference error
double refError(Mat draw, Mat food, int& nextIndex)
{
	Mat drawEdge = cannyThreeCh(draw);
	Mat foodEdge = cannyThreeCh(food);

	vector<vector<Point>> drawSeqContours;
	vector<Vec4i> hierarchyD;

	vector<vector<Point>> foodSeqContours;
	vector<Vec4i> hierarchyF;

	findContours(drawEdge.clone(), drawSeqContours, hierarchyD, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	findContours(foodEdge.clone(), foodSeqContours, hierarchyF, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );

	vector<vector<Point>> disjointContour;

	for(int i = 0 ; i < drawSeqContours.size() ; i++)
		if(drawSeqContours[i].size()>50 && hierarchyD[i][3] != -1)
			disjointContour.push_back(drawSeqContours[i]);

	sort(disjointContour.begin(), disjointContour.end(), compareContourSize);

	int drawPointNum = 0;

	for(int i = 0 ; i < drawSeqContours.size() ; i++)
	{
		drawPointNum += int(drawSeqContours[i].size());
	}

	Mat drawDrawContour = Mat::zeros(draw.size(), CV_32FC3);
	Mat foodDrawContour = Mat::zeros(food.size(), CV_32FC3);

	for(int i = 0 ; i < drawSeqContours.size() ; i++)
		drawContours( drawDrawContour, drawSeqContours, i, Scalar(255, 255, 255), 1, 8);
	
	for(int i = 0 ; i < foodSeqContours.size() ; i++)
		drawContours( foodDrawContour, foodSeqContours, i, Scalar(255, 255, 255), 1, 8);

	//Mat drawConGray;
	Mat foodConGray;

	//cvtColor(drawDrawContour, drawConGray, CV_BGR2GRAY);
	cvtColor(foodDrawContour, foodConGray, CV_BGR2GRAY);

	//Mat drawBin = drawConGray > 128;
	Mat foodBin = foodConGray > 128;

	//Mat nonZeroDraw;
	Mat nonZeroFood;
	//findNonZero(drawBin, nonZeroDraw);
	findNonZero(foodBin, nonZeroFood);
	
	int thrC = 3;
	double score = 0;
	double contourErr = 0;
	vector<double> pointDist;
	vector<double> perContourErr;

	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		for(int j = 0 ; j < disjointContour[i].size() ; j++)
		{
			Point locD = disjointContour[i][j];

			for(int k = 0 ; k < nonZeroFood.rows ; k++)
			{
				Point locF = nonZeroFood.at<Point>(k);
				pointDist.push_back(norm(locF-locD));
			}
			double tmp = *min_element(pointDist.begin(), pointDist.end());
			//cout << tmp << endl;
			contourErr += tmp;
			score += tmp;
			pointDist.clear();
		}
		perContourErr.push_back(contourErr/disjointContour[i].size());
		contourErr = 0;
	}

	//for(int i = 0 ; i < perContourErr.size() ; i++)
	//{
	//	cout <<i<<": "<<perContourErr[i]<<endl;
	//}
	nextIndex = distance(perContourErr.begin(), max_element (perContourErr.begin(), perContourErr.end()));

	//cout << "size: "<< *max_element(perContourErr.begin(), perContourErr.end());


	return score/drawPointNum;
}

//return the min error value fragment of index
int minValueInMap( vector<map<string, int> > input)
{
	vector<map<string, int> >::iterator iterV;

	int returnIndex = 0;
	int minValue = 0;
	string mapKey = "sError";

	for(iterV = input.begin() ; iterV != input.end() ; iterV++)
	{
		if(iterV == input.begin())
		{
			minValue = (*iterV)[mapKey];
			returnIndex =  distance(input.begin(),iterV);
		}
		else
		{
			if((*iterV)[mapKey] < minValue)
			{
				minValue = (*iterV)[mapKey];
				returnIndex = distance(input.begin(), iterV);
			}
		}
	}
	return returnIndex;
}

vector<int> vecSeqIndex(vector<map<string, int> > input)
{
	vector<map<string, int> > copyTmp(input.begin(), input.end()); 
	vector<int> returnSeq;

	int index = 0;

	while(copyTmp.size() != 0)
	{
		index = minValueInMap(copyTmp);
		returnSeq.push_back(index);
		copyTmp.erase(copyTmp.begin()+index);
	}

	return returnSeq;
}

void preProcess(vector<vector<Mat> >* p_foodDesSeq, vector<vector<Point> >* p_sampleResult) {
	
	int idx = 0;
	int count = 0;

	vector<Mat> get_foodDesSeq = vecmatread("preData/foodDesSeq.bin");
	vector<Point> get_sampleResult = vecPointRead("preData/sampleResult.bin");
	vector<Mat> buff_foodDesSeq;
	vector<Point> buff_sampleResult;

	for (int i = 0; i < get_foodDesSeq.size(); i++)
	{
		count++;
		if (count < 70) {
			buff_foodDesSeq.push_back(get_foodDesSeq[i]);
			buff_sampleResult.push_back(get_sampleResult[i]);
		}
		else {
			idx++;
			count = 0;
			p_foodDesSeq->push_back(buff_foodDesSeq);
			p_sampleResult->push_back(buff_sampleResult);
			buff_foodDesSeq.clear();
			buff_sampleResult.clear();
		}
	}
}

vector<Mat> vecmatread(const string& filename)
{
	vector<Mat> matrices;
	ifstream fs(filename, fstream::binary);

	// Get length of file
	fs.seekg(0, fs.end);
	int length = fs.tellg();
	fs.seekg(0, fs.beg);

	while (fs.tellg() < length)
	{
		// Header
		int rows, cols, type, channels;
		fs.read((char*)&rows, sizeof(int));         // rows
		fs.read((char*)&cols, sizeof(int));         // cols
		fs.read((char*)&type, sizeof(int));         // type
		fs.read((char*)&channels, sizeof(int));     // channels

		// Data
		Mat mat(rows, cols, type);
		fs.read((char*)mat.data, CV_ELEM_SIZE(type) * rows * cols);

		matrices.push_back(mat);
	}
	cout << "Preprocess: get food descriptor finish!" << endl;
	return matrices;
}

vector<Point> vecPointRead(const string& filename)
{
	Point buff;
	vector<Point> matrices;
	ifstream fs(filename, fstream::binary);

	int num;
	fs.read((char *)&num, sizeof(num));

	for (int i = 0; i<num; ++i) {
		int val_x, val_y;
		fs.read((char *)&val_x, sizeof(val_x));
		fs.read((char *)&val_y, sizeof(val_y));
		buff.x = val_x;
		buff.y = val_y;
		matrices.push_back(buff);
	}
	return matrices;
}