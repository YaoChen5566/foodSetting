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
double refError(Mat draw, Mat food);

//subPointSeq
vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);

//add image with transparent background
Mat addTransparent(Mat &bg, Mat &fg);

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
/*
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
*/

int main()
{
	
	//Mat draw = imread("inputImg/inin.png", -1);
	//Mat food = imread("foodImg/mouth.png", -1);

	//cout << refError(draw, food)<<endl;



	Mat userDraw = imread("inputImg/inin.png", -1);

	vector<Mat> channels;
	split(userDraw, channels);

	Mat B = channels[0];
	Mat G = channels[1];
	Mat R = channels[2];

	Mat cannyB, cannyG, cannyR;

	Canny(B, cannyB, 50, 150, 3);
	Canny(G, cannyG, 50, 150, 3);
	Canny(R, cannyR, 50, 150, 3);

	Mat cannyColor = cannyThreeCh(userDraw);

	bitwise_or(cannyB, cannyG, cannyColor);
	bitwise_or(cannyColor, cannyR, cannyColor);

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

	Mat drawing = Mat::zeros( userDraw.size(), CV_8UC3 );
	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		cout << disjointContour[i].size()<<endl;;
		drawContours( drawing, disjointContour, i ,  Scalar( 255, 255, 255), 1, 8);
	}

	Mat drawConGray;
	//Mat foodConGray;
	
	cvtColor(drawing, drawConGray, CV_BGR2GRAY);
	
	Mat drawBin = drawConGray > 128;
	//Mat foodBin = foodConGray > 128;

	Mat nonZeroDrawEdge;
	//Mat nonZeroFood;
	findNonZero(drawBin, nonZeroDrawEdge);

	//key nonzeroPointIndex, value: vector save map information
	cfMap pixelCandidate;
	fragList initMap;
	for(int i = 0 ; i < nonZeroDrawEdge.rows ; i++)
	{
		pixelCandidate.Element[i] = initMap;
	}

	imwrite("canny.png", cannyColor);
	imwrite("contour.png", drawing);

	string dir = string("foodImg/");
	//vector<string> files = vector<string>();
	getdir(dir, files);

	cfMap foodCandidate;

	fragList pairSeq;
	clock_t start = clock(); // compare start
	for(int i = 0 ; i < disjointContour.size() ; i++)
	{
		
		descri descriUser(userDrawContours[i]);
		Mat userDrawDes = descriUser.resultDescri();
		//imwrite("_des1.jpg", userDrawDes);
		for(int j = 2 ; j < files.size() ; j++)
		{
			string foodImg = dir + files[j];
			Mat food = imread(foodImg, -1);

			descri desFood(foodImg);
			vector<Mat> foodDesSeq = desFood.seqDescri();
			//imwrite("_des2.jpg", foodDesSeq[0]);
			comp compDes(userDrawDes,foodDesSeq, descriUser.sampleResult(), desFood.sampleResult(), i, j);

			fragList tmpPairSeq; 
			tmpPairSeq.Element = compDes.fragList();

			cout <<"file: "<<files[j]<<endl;

			if(tmpPairSeq.Element.size() > 0)
			{

				for(int k = 0 ; k < tmpPairSeq.Element.size() ; k++)
				{
					cout <<"contour: "<<tmpPairSeq.Element[k]["cIndex"]<<endl;
					cout <<"file: "<<tmpPairSeq.Element[k]["fIndex"]<<endl;
					cout <<"reference index: "<<tmpPairSeq.Element[k]["r"]<<endl;
					cout <<"query index: "<<tmpPairSeq.Element[k]["q"]<<endl;
					cout <<"match length: "<<tmpPairSeq.Element[k]["l"]<<endl;

					//warping and save the information for pixel and fragment
					vector<Point> matchSeq1 = subPointSeq(descriUser.sampleResult(), tmpPairSeq.Element[k]["r"], tmpPairSeq.Element[k]["l"]);
					vector<Point> matchSeq2 = subPointSeq(desFood.sampleResult(), tmpPairSeq.Element[k]["q"], tmpPairSeq.Element[k]["l"]);

					Mat foodDrawContour = Mat::zeros(food.size(), CV_32FC4);
					vector< vector<Point> > tmpContour;
					tmpContour.push_back(desFood.sampleResult());

					for(int m = 0 ; m < tmpContour.size() ; m++)
						drawContours( foodDrawContour, tmpContour, m, Scalar(255, 255, 255, 255), 1, 8);

					Mat warp_mat = estimateRigidTransform(matchSeq2, matchSeq1, false); //(src, dst)

					Mat drawClone = userDraw.clone();
					warpAffine(foodDrawContour, drawClone, warp_mat, drawClone.size());
					
					Mat foodDrawContourBin = drawConGray > 128;
					
					Mat nonZeroFoodDraw;
					findNonZero(drawBin, nonZeroFoodDraw);

					for(int a = 0 ; a < nonZeroDrawEdge.rows ; a++)
					{
						Point locD = nonZeroDrawEdge.at<Point>(a);
						for(int b = 0 ; b < nonZeroFoodDraw.rows ; b++)
						{
							Point locF = nonZeroFoodDraw.at<Point>(b);
							if (norm(locF-locD) == 0)
							{
								fragList tmpFL;
								tmpFL.Element.push_back(tmpPairSeq.Element[k]);
								pixelCandidate.Element[a] = tmpFL;
							}
						}
					}
					//warpAffine(food, userDraw2, rot_mat, userDraw.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
					//Mat result = addTransparent(userDraw, drawClone);
				}
			}
			pairSeq.Element.insert(pairSeq.Element.end(), tmpPairSeq.Element.begin(), tmpPairSeq.Element.end());
		}
		foodCandidate.Element[i] = pairSeq;	
	}

	clock_t finish = clock(); // compare finish

	cout << "time: " << finish-start<<endl;

	//	cfMap foodCandidate; fragList pairSeq;

//	clock_t start = clock(); // compare start

	tree<map<string, int> > stackTree;
	tree<map<string, int> >::iterator root;
	tree<map<string, int> >::iterator child;

	root = stackTree.begin();

	map<int, fragList>::iterator iter1;
	vector<map<string, int> >::iterator iter2;

	// first layer
	for(iter1 = foodCandidate.Element.begin() ; iter1 != foodCandidate.Element.end() ; iter1++)
	{
		for(iter2 = iter1->second.Element.begin() ; iter2 != iter1->second.Element.end() ; iter2++)
		{
			//cout << (*iter2)["r"]<<endl;
			//map<string, int> tmpMap = (*iter2);
			//child = stackTree.append_child(root, (*iter2));
		}
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

	vector<map<string, int>> tmpppp = compDes.fragList();
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
	clock_t start = clock(); // compare start

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
	clock_t finish = clock(); // compare finish

	cout << "time: " << finish-start<<endl;

	/*for(int i = 0 ; i < nonZeroFood.rows ; i++)
	{
		Point locF = nonZeroFood.at<Point>(i);

		for(int j = 0 ; j < nonZeroDraw.rows ; j++)
		{
			Point locD = nonZeroDraw.at<Point>(j);
			//if(norm(locF-locD)<1)
				//cout << norm(locF-locD)<<endl;
			pointDist.push_back(norm(locF-locD));
		}
		double tmp = *min_element(pointDist.begin(), pointDist.end());
		cout << tmp << endl;
		score += tmp;
		pointDist.clear();
	}*/

	//cout << "Non-Zero Locations = " << nonZeroDraw << endl << endl;

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
			//cout << orValue<<endl;
			if(orValue == 1)
			{
				Vec4b pixDraw = draw.at<Vec4b>(i, j);
				Vec4b pixFood = food.at<Vec4b>(i, j);

				tmp =  sqrt(pow(pixDraw.val[0]-pixFood.val[0], 2) + pow(pixDraw.val[1]-pixFood.val[1], 2) + pow(pixDraw.val[2]-pixFood.val[2], 2));
				cout << score << endl;

				score += tmp;

			}
		}
	}

	return score/(draw.rows*draw.cols);
}

//reference error
double refError(Mat draw, Mat food)
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

	cout <<"size: "<<drawSeqContours.size()<<endl;

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

	/*
	for(int i = 0 ; i < nonZeroDraw.rows ; i++)
	{
		Point locD = nonZeroDraw.at<Point>(i);

		for(int j = 0 ; j < nonZeroFood.rows ; j++)
		{
			Point locF = nonZeroFood.at<Point>(j);
			double dist = norm(locF-locD);
			
			if(dist < thrC)
				pointNum++;
		}
	}
	*/
	for(int i = 0 ; i < perContourErr.size() ; i++)
	{
		cout <<i<<": "<<perContourErr[i]<<endl;
	}
	return score/drawPointNum;
}