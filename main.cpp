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


struct fragList
{
	vector<map<string, int> > Element;
};

struct cfMap
{
	map<int, fragList> Element;
};

//all files(food image) in the dir
vector<string> files = vector<string>();

//single test
void singleTest(void);
//warp test
void warpTest(void);
//print vector


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

//gradient error 
double gradError(Mat draw, Mat food);

//subPointSeq
vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);

//add image with transparent background
Mat addTransparent(Mat &bg, Mat &fg);

// return the min error value index of fragment
int minValueInMap( vector<map<string, int> > input);

//return the vector of seqence of error value from small to large
vector<map<string, int> > vecSeqIndex(vector<map<string, int> > input);

//return total error value
double getTotalErr(int state, int& nextIndex, int& nextFrag, double& refErr, vector<vector<Point> >& samplepointsOfDraw, vector<vector<Point> >& samplepointsOfFood, vector<fragList>& sortedFragList, Mat& resultStack, Mat& resultStackClone, string& dir, Mat& userDraw);

vector<string> split_str(string s, char ch);

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

bool compareWithCertainKey(map<string, int> input1, map<string, int> input2)
{
	int i = input1["sError"];
	int j = input2["sError"];
	return(i<j);
}

// print tree
void print_tree(const tree<std::string>& tr, tree<std::string>::pre_order_iterator it, tree<std::string>::pre_order_iterator end)
{
	if(!tr.is_valid(it)) return;
	int rootdepth=tr.depth(it);
	std::cout << "-----" << std::endl;
	while(it!=end) {
		for(int i=0; i<tr.depth(it)-rootdepth; ++i) 
			std::cout << "  ";
		std::cout << (*it) << std::endl << std::flush;
		++it;
		}
	std::cout << "-----" << std::endl;
}




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
		if(userDrawContours[i].size()>70 && hierarchy[i][3] != -1)
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
		//cout << files[76+2]<<endl;
		cout <<"contour index: "<<i<<", contour size: "<< disjointContour[i].size()<<endl;
		

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
					//cout << warp_mat.size()<<endl;

					if(warp_mat.size() == cv::Size(0, 0))
					{
						tmpPairSeq.Element.erase (tmpPairSeq.Element.begin()+k);
					}
					else
					{
						//Mat drawClone = userDraw.clone();
						warpAffine(food, foodStack, warp_mat, foodStack.size());

						int tmpp;
						tmpPairSeq.Element[k]["eError"] = edgeError(userDraw, foodStack);
						tmpPairSeq.Element[k]["cError"] = colorError(userDraw, foodStack);
						tmpPairSeq.Element[k]["rError"] = refError(userDraw, foodStack, tmpp);
						tmpPairSeq.Element[k]["sError"] = tmpPairSeq.Element[k]["eError"] + tmpPairSeq.Element[k]["cError"] + tmpPairSeq.Element[k]["rError"];
					}
				}
			}
			pairSeq.Element.insert(pairSeq.Element.end(), tmpPairSeq.Element.begin(), tmpPairSeq.Element.end());
		}
		foodCandidate.Element[i] = pairSeq;	
		pairSeq.Element.clear();
		

		//cout <<i<<": " <<pairSeq.Element.size()<<endl;
	}
	
	for(int i = 0 ; i < disjointContour.size() ; i++)
		cout <<i<<"'s candidate" <<foodCandidate.Element[i].Element.size()<<endl;


	
	int nextIndex = 0, nextFrag = 0, fragPtr = 0, preIndex;
	double totalErr, preErr = 100000000;
	double refErr = 0;
	Mat resultStack, resultStackClone;
	int stackState = 0;
	vector<string> cfSeq;
	vector<int> contourVec, fragVec;
	vector<double> errSeq;
	vector<double> leafErr;
	vector<vector<int> > contourMatchSeq;
	tree<string> tr;
	tree<string>::iterator root, findLoc;
	vector<fragList> sortedFragList;
	bool finish = false;
	
	for (int i = 0; i < disjointContour.size(); i++)
	{
		fragList tmpFragList;
		tmpFragList.Element = vecSeqIndex(foodCandidate.Element[i].Element);
		sortedFragList.push_back(tmpFragList);
	}

	contourVec.push_back(-1);
	fragVec.push_back(-1);

	//nextFrag = contourMatchSeq[nextIndex][0];

	
	//for(int i = 0 ; i < sortedFragList.size() ; i++)
	//{
	//	for(int j = 0 ; j < sortedFragList[i].Element.size() ; j++)
	//	{
	//		cout << sortedFragList[i].Element[j]["cIndex"] << " ";
	//	}
	//	cout << endl;
	//}

	errSeq.push_back(preErr);

	preIndex = nextIndex;

	clock_t start = clock(); // compare start


	while(1)
	{
		
		totalErr = getTotalErr(stackState, nextIndex, nextFrag, refErr, samplepointsOfDraw, samplepointsOfFood, sortedFragList, resultStack, resultStackClone, dir, userDraw);
		cout <<"preIndex: "<<preIndex<<" , nextIndex: "<<nextIndex<<endl;
		//cout <<"fragIndex: "<<nextFrag<<endl; 
		cout << "preErr: " << errSeq.back() << ", totalErr: " << totalErr << endl;
		if(errSeq.back() > refErr && (errSeq.back()-refErr) > 0.1*errSeq.back())
		{
			contourVec.push_back(preIndex);
			fragVec.push_back(nextFrag);
			if(stackState == 0)
			{
				cfSeq.push_back(to_string(contourVec.back())+"*"+to_string(fragVec.back()));
				root = tr.begin();
				findLoc=tr.insert(root, cfSeq.back());
				stackState = 1;
			}
			else
			{
				findLoc = find(tr.begin(), tr.end(), cfSeq.back());
				cfSeq.push_back(cfSeq.back()+"_"+to_string(contourVec.back())+"*"+to_string(fragVec.back()));
				tr.append_child(findLoc, cfSeq.back());
			}
			preIndex = nextIndex;
			errSeq.push_back(refErr);
			nextFrag = 0;
		}
		else
		{
			findLoc = find(tr.begin(), tr.end(), cfSeq.back());
			cfSeq.push_back(cfSeq.back()+"_"+to_string(preIndex)+"*"+to_string(nextFrag));
			tr.append_child(findLoc, cfSeq.back());
			findLoc = find(tr.begin(), tr.end(), cfSeq.back());
			tr.append_child(findLoc, to_string(totalErr));
			leafErr.push_back(totalErr);
			
			cfSeq.pop_back();
			resultStack = resultStackClone.clone();
			
			nextFrag++;
			while(nextFrag >= sortedFragList[preIndex].Element.size())
			{
				if(contourVec.back() == 0)
				{
					stackState = 0;
					finish = true;
					break;
				}
				else
				{
					nextIndex = contourVec.back();
					nextFrag = fragVec.back()+1;

					contourVec.pop_back();
					fragVec.pop_back();
					errSeq.pop_back();
					cfSeq.pop_back();

					preErr = errSeq.back();
					preIndex = nextIndex;
					cout <<"!!!!!"<<preIndex<<"~~~~~"<<nextIndex<<endl;
					
				}
			}
			
			nextIndex = preIndex;
			
			
		
		}
		//print_tree(tr, tr.begin(), tr.end());
		if(finish)
			break;
	}

	cout <<"!!!"<<endl;

	sort(leafErr.begin(), leafErr.end());

	int numOfResult = 10;

	for(int i = 0 ; i < numOfResult ; i++)
	{
		tree<string>::iterator iter;
		iter = find(tr.begin(), tr.end() , to_string(leafErr[i]));
		iter--;

		vector<string> cfList = split_str((*iter), '_');
		
		Mat resultStack = userDraw.clone();
			
		for(vector<string>::iterator cf = cfList.begin() ; cf != cfList.end() ; cf++)
		{
			vector<string> cAndf = split_str((*cf), '*');
			int stackC = stoi(cAndf[0]);
			int stackF = stoi(cAndf[1]);
			//cout << stackC<<" "<<stackF<<endl;

			//get contour and food subsequence
			vector<Point> matchSeqDraw = subPointSeq(samplepointsOfDraw[sortedFragList[stackC].Element[stackF]["cIndex"]], sortedFragList[stackC].Element[stackF]["r"], sortedFragList[stackC].Element[stackF]["l"]);
			vector<Point> matchSeqFood = subPointSeq(samplepointsOfFood[sortedFragList[stackC].Element[stackF]["fIndex"]], sortedFragList[stackC].Element[stackF]["q"], sortedFragList[stackC].Element[stackF]["l"]);

			//get warping matrix
			Mat warpMat_2 = estimateRigidTransform(matchSeqFood, matchSeqDraw, false); //(src, dst)

			Mat resultStack_2 = resultStack.clone();
			warpAffine(imread(dir + files[sortedFragList[stackC].Element[stackF]["fIndex"] + 2], -1), resultStack_2, warpMat_2, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
			//resultStackClone = resultStack.clone();
			resultStack = addTransparent(resultStack, resultStack_2);

		}
		imwrite("result_"+to_string(i)+".png", resultStack);


		//cout <<(*iter)<<endl;
	}
	clock_t finishT = clock(); // compare finish

	cout << "time: " << finishT-start<<endl;

	//singleTest();
	system("Pause");
}

//single test
void singleTest(void)
{
	clock_t start = clock(); // compare start
	string tmp = "foodImg/137.png";
	string tmp2 = "foodImg/mouth.png";
	Mat input1 = imread(tmp, -1);
	Mat input2 = imread(tmp2, -1);

	descri descri1(tmp);
	Mat inputDes1 = descri1.resultDescri();
	descri descri2(tmp2);
	Mat inputDes2 = descri2.resultDescri();
	vector<Mat> inputDesSeq2 = descri2.seqDescri();

	//cout << inputDesSeq2.size()<<endl;;

	//comp compDes(inputDes1,inputDes2);
	comp compDes(inputDes1, inputDesSeq2, descri1.sampleResult(), descri2.sampleResult(), 0, 0);

	vector<map<string, int> > tmpppp = compDes.fragList();

	cout <<"fragSize: "<< tmpppp.size()<<endl;;

	clock_t finish = clock(); // compare finish

	int vecIndex = minValueInMap(tmpppp);

	cout << "time: " << finish-start<<endl;
	cout << "size: " << compDes.fragList().size()<<endl;
	cout << "@start1: "<< tmpppp[1]["r"] <<" @start2: "<< tmpppp[1]["q"] <<" @range: "<< tmpppp[1]["l"] <<endl;

	Mat input1_draw = input1.clone();
	Mat input2_draw = input2.clone();

	Mat warpingResult = input1.clone();

	vector<Point> pointSeq1 = descri1.sampleResult(); 
	vector<Point> pointSeq2 = descri2.sampleResult();

	for(int i = 0 ; i < compDes.range() ; i++)
	{
		circle(input1_draw, pointSeq1[(tmpppp[1]["r"]+i)%pointSeq1.size()],1,Scalar(0,0,255,255),2);
		circle(input2_draw, pointSeq2[(tmpppp[1]["q"]+i)%pointSeq2.size()],1,Scalar(0,0,255,255),2);	
	}


	
	imwrite("result1.png", input1_draw);
	imwrite("result2.png", input2_draw);


	//vector<map<string, int>>::iterator iterV;
	//map<string, int>::iterator iterM;


	vector<Point> matchSeq1 = subPointSeq(pointSeq1, tmpppp[1]["r"], tmpppp[0]["l"]);
	vector<Point> matchSeq2 = subPointSeq(pointSeq2, tmpppp[1]["q"], tmpppp[0]["l"]);

	//for(int i = 0 ; i < matchSeq1.size() ; i++)
	//{
	//	cout <<"1: "<<matchSeq1[i]<<" 2: "<<matchSeq2[i]<<endl;
	//}


	Mat warp_mat = estimateRigidTransform(matchSeq1, matchSeq2, false); //(src, dst)


	cout <<"type: "<<warp_mat.size()<<endl;
	cout <<"scale: "<< pow(warp_mat.at<double>(0,0), 2) + pow(warp_mat.at<double>(1,0), 2)  <<endl;
	
	warpAffine(input1, warpingResult, warp_mat, warpingResult.size());
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
		if(drawSeqContours[i].size()>70 && hierarchyD[i][3] != -1)
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

	nextIndex = distance(perContourErr.begin(), max_element (perContourErr.begin(), perContourErr.end()));

	return score/drawPointNum;
}

//gradient error 
double gradError(Mat draw, Mat food)
{
	Mat blurDraw;
	Mat blurFood;

	GaussianBlur(draw, blurDraw, Size(3, 3), 0, 0);
	GaussianBlur(food, blurFood, Size(3, 3), 0, 0);

	Mat gradDraw = cannyThreeCh(blurDraw);
	Mat gradFood = cannyThreeCh(blurFood);

	Mat subResult;
	absdiff(gradDraw, gradFood, subResult);

	return sum(subResult).val[0]/(subResult.cols*subResult.rows);


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

//return total error value
double getTotalErr(int state, int& nextIndex, int& nextFrag, double& refErr, vector<vector<Point> >& samplepointsOfDraw, vector<vector<Point> >& samplepointsOfFood, vector<fragList>& sortedFragList, Mat& resultStack, Mat& resultStackClone, string& dir, Mat& userDraw) {

	//get contour and food subsequence
	vector<Point> matchSeqDraw = subPointSeq(samplepointsOfDraw[sortedFragList[nextIndex].Element[nextFrag]["cIndex"]], sortedFragList[nextIndex].Element[nextFrag]["r"], sortedFragList[nextIndex].Element[nextFrag]["l"]);
	vector<Point> matchSeqFood = subPointSeq(samplepointsOfFood[sortedFragList[nextIndex].Element[nextFrag]["fIndex"]], sortedFragList[nextIndex].Element[nextFrag]["q"], sortedFragList[nextIndex].Element[nextFrag]["l"]);

	//get warping matrix
	Mat warpMat_2 = estimateRigidTransform(matchSeqFood, matchSeqDraw, false); //(src, dst)

	if (state == 0) {
		resultStack = userDraw.clone();
		warpAffine(imread(dir + files[sortedFragList[nextIndex].Element[nextFrag]["fIndex"] + 2], -1), resultStack, warpMat_2, resultStack.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	}
	else {
		Mat resultStack_2 = resultStack.clone();
		warpAffine(imread(dir + files[sortedFragList[nextIndex].Element[nextFrag]["fIndex"] + 2], -1), resultStack_2, warpMat_2, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
		resultStackClone = resultStack.clone();
		resultStack = addTransparent(resultStack, resultStack_2);
	}
	refErr = refError(userDraw, resultStack, nextIndex);
	double totalErr = edgeError(userDraw, resultStack) + colorError(userDraw, resultStack) + refErr;
	return totalErr;
}

vector<map<string, int>> vecSeqIndex(vector<map<string, int> > input)
{
	vector<map<string, int> > copyTmp(input.begin(), input.end()); 
	sort(copyTmp.begin(), copyTmp.end(), compareWithCertainKey);

	return copyTmp;
}

vector<string> split_str(string s, char ch)
{
	vector<string> tokens;
	istringstream ss(s);
	string token;

	while (std::getline(ss, token, ch)) {
		tokens.push_back(token);
	}
	return tokens;
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

