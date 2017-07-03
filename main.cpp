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

#include "descri.h"
#include "compare.h"
#include "fragment.h"
#include "recipe.h"
#include "tpSort.h"

#include "tree.hh"


# define PI 3.1415926

using namespace std;
using namespace cv;


struct fragList
{
	vector<frag> Element;
};

struct cfMap
{
	map<int, fragList> Element;
};

//all files(food image) in the dir
vector<string> files = vector<string>();

//single test
void singleTest(void);
//get test img
void testImg(void);
//warp test
void warpTest(void);
// err test
void errTest(void);
// recipe test
void recipeTest(void);
// topo test
void topoTest(void);

//get dir files
int getdir(string dir, vector<string> &files);

//segment image with alpha value
Mat alphaBinary(Mat input);

//three channels Canny
Mat cannyThreeCh(Mat input, bool mode);

//canny alpha channel
Mat cannyAlpha(Mat input);

// edge error
double edgeError(Mat draw, Mat food);

//color error
double colorError(Mat draw, Mat food);

//reference error
double refError(Mat draw, Mat food, int& nextIndex);

//intersection error
double interError(Mat draw, Mat food);

//gradient error 
double gradError(Mat draw, Mat food);

//iError with two contour
double iContourError(vector<Point> contour1, vector<Point> contour2, Mat food);

//subPointSeq
vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int range);

//add image with transparent background
Mat addTransparent(Mat &bg, Mat &fg);

// return the min error value index of fragment
int minValueInMap( vector<map<string, int> > input);

//return the vector of seqence of error value from small to large
fragList vecSeqIndex(fragList input);

//return total error value
double getTotalErr(int state, int& nextIndex, int& nextFrag, double& refErr, vector<vector<Point> >& samplepointsOfDraw, vector<vector<Point> >& samplepointsOfFood, vector<fragList>& sortedFragList, Mat& resultStack, Mat& resultStackClone, Mat& stackEdge, Mat& stackEdgeClone, string& dir, Mat& userDraw);

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

bool compareWithCertainKey(frag input1, frag input2)
{
	double i = input1.iError;
	double j = input2.iError;
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
	
	////Mat userDraw = imread("test/star.png", -1);
	//Mat userDraw = imread("inputImg/inin.png", -1);

	//Mat cannyColor = cannyThreeCh(userDraw, true);

	//vector<vector<Point> > userDrawContours;
	//vector<Vec4i> hierarchy;

	//findContours(cannyColor.clone(), userDrawContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
	//
	//vector<vector<Point> > disjointContour;

	//for(int i = 0 ; i < userDrawContours.size() ; i++)
	//{
	//	if(userDrawContours[i].size()>50 && hierarchy[i][3] != -1)
	//		disjointContour.push_back(userDrawContours[i]);
	//}

	//sort(disjointContour.begin(), disjointContour.end(), compareContourSize);

	//RNG rng(12345);
	//Mat drawing = Mat::zeros( userDraw.size(), CV_8UC3 );
	//for(int i = 0 ; i < disjointContour.size() ; i++)
	//{
	//	cout << disjointContour[i].size()<<endl;
	//	Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
	//	drawContours( drawing, disjointContour, i ,  color, 1, 8);
	//}

	//imwrite("canny.png", cannyColor);
	//imwrite("contour.png", drawing);

	//string dir = string("foodImg/");
	////vector<string> files = vector<string>();
	//getdir(dir, files);

	//// pre-process for contour descriptor and sample points
	//cout <<"start pre-process"<<endl;
	//
	//vector<vector<Mat> > desOfDraw;
	//vector<vector<Point> > samplepointsOfDraw;
	//for(int i = 0 ; i < disjointContour.size() ; i++)
	//{
	//	descri descriUser(disjointContour[i]);
	//	//cout << descriUser.seqDescri().size()<<endl;

	//	desOfDraw.push_back(descriUser.seqDescri());
	//	samplepointsOfDraw.push_back(descriUser.sampleResult());
	//}
	////pre-Process for food descriptor and sample points
	//vector<vector<Mat> > desOfFood;
	//vector<vector<Point> > samplepointsOfFood;
	//preProcess(&desOfFood, &samplepointsOfFood);

	//cout << "pre-process done"<<endl;

	//// read recipe
	//recipe recIn;
	//recIn.readRecipeFile("duck.bin");

	//int fragNum = 0;

	//cfMap foodCandidate;

	//fragList pairSeq;

	//for(int i = 0 ; i < desOfDraw.size() ; i++)
	//{
	//	clock_t start = clock(); // compare start
	//	//cout << files[126+2]<<endl;
	//	cout <<"contour index: "<<i<<", contour size: "<< disjointContour[i].size()<<endl;
	//	
	//	for(int j = 2 ; j < files.size() ; j++)
	//	{
	//		cout <<"file: "<<files[j]<<endl;

	//		comp compDes(desOfDraw[i], desOfFood[j-2], samplepointsOfDraw[i], samplepointsOfFood[j-2], i, j, userDraw.size());
	//		//comp compDes(desOfDraw[i], desFood.seqDescri(), samplepointsOfDraw[i], desFood.sampleResult(), i, j);

	//		fragList tmpPairSeq; 
	//		tmpPairSeq.Element = compDes.fragList2();

	//		//cout << tmpPairSeq.Element.size()<<endl;

	//		if(tmpPairSeq.Element.size() > 0)
	//		{
	//			fragNum += tmpPairSeq.Element.size();

	//			for(int k = 0 ; k < tmpPairSeq.Element.size() ; k++)
	//			{
	//				/*cout <<"contour: "<<tmpPairSeq.Element[k].cIndex<<endl;
	//				cout <<"file: "<<files[tmpPairSeq.Element[k].fIndex+2]<<endl;
	//				cout <<"reference index: "<<tmpPairSeq.Element[k].r<<endl;
	//				cout <<"query index: "<<tmpPairSeq.Element[k].q<<endl;
	//				cout <<"match length: "<<tmpPairSeq.Element[k].l<<endl;*/

	//				string foodImg = dir + files[j];
	//				Mat food = imread(foodImg, -1);
	//				Mat foodEdge = Mat::zeros(userDraw.size(), CV_8UC4);
	//				Mat foodEdgeAffine = Mat::zeros(userDraw.size(), CV_8UC4);
	//				Mat foodStack = Mat::zeros(userDraw.size(), CV_8UC4);
	//				Mat drawClone = Mat::zeros(userDraw.size(), CV_8UC4);

	//				drawContours( foodEdge, vector<vector<Point> >(1, samplepointsOfFood[j]), 0 ,  Scalar(255,0,0,255), 1, 8);

	//				//cout << tmpPairSeq.Element[k].warpMatrix.size()<<endl;

	//				Mat addUF = addTransparent(userDraw, tmpPairSeq.Element[k].warpResult);

	//				imwrite("testI/"+to_string(i)+"_"+to_string(j)+"_"+to_string(tmpPairSeq.Element[k].iError)+".png",addUF);

	//				int tmpp;
	//				//tmpPairSeq.Element[k].setError(edgeError(userDraw, foodEdgeAffine), colorError(userDraw,foodStack), refError(userDraw, foodEdgeAffine, tmpp), tmpPairSeq.Element[k].iError, tmpPairSeq.Element[k].iErrorRatio1);

	//			}
	//		}
	//		pairSeq.Element.insert(pairSeq.Element.end(), tmpPairSeq.Element.begin(), tmpPairSeq.Element.end());
	//	}
	//	foodCandidate.Element[i] = pairSeq;	
	//	pairSeq.Element.clear();
	//	clock_t finishT = clock(); // compare finish
	//	cout << "time: " << finishT-start<<endl;

	//	//cout <<i<<": " <<pairSeq.Element.size()<<endl;
	//}
	//
	//for(int i = 0 ; i < disjointContour.size() ; i++)
	//	cout <<i<<"'s candidate" <<foodCandidate.Element[i].Element.size()<<endl;

	//vector<fragList> sortedFragList;

	//for (int i = 0; i < disjointContour.size() ; i++)
	//{
	//	fragList tmpFragList;
	//	sort(foodCandidate.Element[i].Element.begin(), foodCandidate.Element[i].Element.end(), compareWithCertainKey);
	//	sortedFragList.push_back(foodCandidate.Element[i]);
	//	//sortedFragList[i].Element.erase(sortedFragList[i].Element.begin()+2, sortedFragList[i].Element.end());
	//}

	//// topological sort
	//topo graph((int) disjointContour.size());

	//for(int i = 0 ; i < disjointContour.size() ; i++)
	//{
	//	for(int j = i+1 ; j < disjointContour.size() ; j++)
	//	{
	//		Mat iContourImg = sortedFragList[i].Element[0].warpResult.clone();
	//		Mat jContourImg = sortedFragList[j].Element[0].warpResult.clone();

	//		vector<Point> iContour, jContour;
	//		iContour.assign(samplepointsOfFood[sortedFragList[i].Element[0].fIndex-2].begin(), samplepointsOfFood[sortedFragList[i].Element[0].fIndex-2].end());
	//		jContour.assign(samplepointsOfFood[sortedFragList[j].Element[0].fIndex-2].begin(), samplepointsOfFood[sortedFragList[j].Element[0].fIndex-2].end());

	//		//draw contour i first fragment area after warping
	//		for(int ic = 0 ; ic < iContourImg.cols ; ic++)
	//		{
	//			for(int ir = 0 ; ir < iContourImg.rows ; ir++)
	//			{
	//				Vec4b BGRA = iContourImg.at<Vec4b>(ir, ic);
	//				Vec4b newBGRA = Vec4b(0, 255, 0, 255);
	//				if(BGRA[3] != 0)
	//					iContourImg.at<Vec4b>(ir, ic) = newBGRA;
	//			}
	//		}

	//		//draw contour j first fragment area after warping
	//		for(int jc = 0 ; jc < jContourImg.cols ; jc++)
	//		{
	//			for(int jr = 0 ; jr < jContourImg.rows ; jr++)
	//			{
	//				Vec4b BGRA = jContourImg.at<Vec4b>(jr, jc);
	//				Vec4b newBGRA = Vec4b(255, 0, 0, 255);
	//				if(BGRA[3] != 0)
	//					jContourImg.at<Vec4b>(jr, jc) = newBGRA;
	//			}
	//		}

	//		//stack j on i
	//		Mat cannyStackJI = cannyThreeCh( addTransparent(iContourImg, jContourImg), true);
	//		//stack i on j
	//		Mat cannyStackIJ = cannyThreeCh( addTransparent(jContourImg, iContourImg), true);

	//		double joni = iContourError(samplepointsOfDraw[i], samplepointsOfDraw[j], cannyStackJI);
	//		double ionj = iContourError(samplepointsOfDraw[i], samplepointsOfDraw[j], cannyStackIJ);

	//		imwrite("i.png", iContourImg);
	//		imwrite("j.png", jContourImg);
	//		imwrite("joni"+to_string(joni)+".png", cannyStackJI);
	//		imwrite("ionj"+to_string(ionj)+".png", cannyStackIJ);

	//		//cout << j<<" on "<<i<<" : " <<joni<<endl;
	//		//cout << i<<" on "<<j<<" : " <<ionj<<endl;

	//		//smaller cannyStackJI, add edge ij
	//		if(joni < ionj)
	//			graph.addEdge(i, j);
	//		else if(ionj < joni)
	//			graph.addEdge(j, i);

	//		Mat drawConIJ = Mat::zeros(userDraw.size(), CV_8UC3);

	//		drawContours(drawConIJ, samplepointsOfDraw, i, Scalar(255, 255, 255), 1);
	//		drawContours(drawConIJ, samplepointsOfDraw, j, Scalar(255, 255, 255), 1);

	//	}
	//}

	//graph.printAdjList();
	//graph.topoSort();

	//cout <<"size: "<<graph.sortResult.size()<<endl;
	//for(int i = 0 ; i < graph.sortResult.size() ; i++)
	//	cout << graph.sortResult[i]<<" ";


	//for(int i = 0 ; i < disjointContour.size() ; i++)
	//{
	//	bool needComp = true;
	//	for(int j = 0 ; j < foodCandidate.Element[i].Element.size() ; j++)
	//	{
	//		if(foodCandidate.Element[i].Element[j].iError <= 2)
	//		{
	//			needComp = false;
	//			break;
	//		}
	//	}
	//	cout << needComp<<endl;
	//}

	//
	//int nextIndex = 0, nextFrag = 0, fragPtr = 0, preIndex;
	//double totalErr, preErr = 100000000;
	//double refErr = 0;
	//Mat resultStack, resultStackClone;
	//Mat stackEdge, stackEdgeClone;
	//int stackState = 0;
	//vector<string> cfSeq;
	//vector<int> contourVec, fragVec;
	//vector<double> errSeq;
	//vector<double> leafErr;
	//vector<vector<int> > contourMatchSeq;
	//tree<string> tr;
	//tree<string>::iterator root, findLoc;
	//vector<fragList> sortedFragList;
	//bool finish = false;
	
	//}
	//	
	//for(int i = 0 ; i < sortedFragList.size() ; i++)
	//{
	//	for(int j = 0 ; j < sortedFragList[i].Element.size() ; j++)
	//	{
	//		cout << sortedFragList[i].Element[j].sError << " ";
	//	}
	//	cout << endl;
	//}

	
	/*
	for(int i = 0 ; i < 3 ; i++)
	{
		for(int j = 0 ; j < 3 ; j++)
		{
			for(int m = 0 ; m < 3 ; m++)
			{
				for(int n = 0 ; n < 3 ; n++)
				{
					cout <<"i: "<<i<<", j: "<<j<<", m: "<<m<<", n: "<<n<<endl;
					resultStack = Mat::zeros(userDraw.size(), CV_8UC4);
					Mat resultStack_2 = resultStack.clone();
					warpAffine(imread(dir+files[sortedFragList[0].Element[i].fIndex + 2], -1), resultStack, sortedFragList[0].Element[i].warpMatrix, resultStack.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
					warpAffine(imread(dir+files[sortedFragList[1].Element[j].fIndex + 2], -1), resultStack_2, sortedFragList[1].Element[j].warpMatrix, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
					resultStack = addTransparent(resultStack, resultStack_2);
					warpAffine(imread(dir+files[sortedFragList[2].Element[m].fIndex + 2], -1), resultStack_2, sortedFragList[2].Element[m].warpMatrix, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
					resultStack = addTransparent(resultStack, resultStack_2);
					warpAffine(imread(dir+files[sortedFragList[3].Element[n].fIndex + 2], -1), resultStack_2, sortedFragList[3].Element[n].warpMatrix, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
					resultStack = addTransparent(resultStack, resultStack_2);
					imwrite("testR/"+to_string(i)+to_string(j)+to_string(m)+to_string(n)+".png", resultStack);
				}
			}
		}
	}*/

	//contourVec.push_back(-1);
	//fragVec.push_back(-1);

	////nextFrag = contourMatchSeq[nextIndex][0];

	//errSeq.push_back(preErr);

	//preIndex = nextIndex;

	//while(1)
	//{
	//	
	//	totalErr = getTotalErr(stackState, nextIndex, nextFrag, refErr, samplepointsOfDraw, samplepointsOfFood, sortedFragList, resultStack, resultStackClone, stackEdge, stackEdgeClone, dir, userDraw);
	//	cout <<"preIndex: "<<preIndex<<" , nextIndex: "<<nextIndex<<endl;
	//	//cout <<"fragIndex: "<<nextFrag<<endl; 
	//	cout << "preErr: " << errSeq.back() << ", totalErr: " << totalErr << endl;
	//	if(errSeq.back() > refErr && (errSeq.back()-refErr) > 0.1*errSeq.back())
	//	{
	//		contourVec.push_back(preIndex);
	//		fragVec.push_back(nextFrag);
	//		if(stackState == 0)
	//		{
	//			cfSeq.push_back(to_string(contourVec.back())+"*"+to_string(fragVec.back()));
	//			root = tr.begin();
	//			findLoc=tr.insert(root, cfSeq.back());
	//			stackState = 1;
	//		}
	//		else
	//		{
	//			findLoc = find(tr.begin(), tr.end(), cfSeq.back());
	//			cfSeq.push_back(cfSeq.back()+"_"+to_string(contourVec.back())+"*"+to_string(fragVec.back()));
	//			tr.append_child(findLoc, cfSeq.back());
	//		}
	//		preIndex = nextIndex;
	//		errSeq.push_back(refErr);
	//		nextFrag = 0;
	//	}
	//	else
	//	{
	//		findLoc = find(tr.begin(), tr.end(), cfSeq.back());
	//		cfSeq.push_back(cfSeq.back()+"_"+to_string(preIndex)+"*"+to_string(nextFrag));
	//		tr.append_child(findLoc, cfSeq.back());
	//		findLoc = find(tr.begin(), tr.end(), cfSeq.back());
	//		tr.append_child(findLoc, to_string(totalErr));
	//		leafErr.push_back(totalErr);
	//		
	//		cfSeq.pop_back();
	//		resultStack = resultStackClone.clone();
	//		
	//		nextFrag++;
	//		while(nextFrag >= sortedFragList[preIndex].Element.size())
	//		{
	//			if(contourVec.back() == 0)
	//			{
	//				stackState = 0;
	//				finish = true;
	//				break;
	//			}
	//			else
	//			{
	//				nextIndex = contourVec.back();     
	//				nextFrag = fragVec.back()+1;

	//				contourVec.pop_back();
	//				fragVec.pop_back();
	//				errSeq.pop_back();
	//				cfSeq.pop_back();

	//				preErr = errSeq.back();
	//				preIndex = nextIndex;
	//				cout <<"!!!!!"<<preIndex<<"~~~~~"<<nextIndex<<endl;
	//				
	//			}
	//		}
	//		
	//		nextIndex = preIndex;
	//		
	//		
	//	
	//	}
	//	print_tree(tr, tr.begin(), tr.end());
	//	if(finish)
	//		break;
	//}

	//cout <<"!!!"<<endl;

	//sort(leafErr.begin(), leafErr.end());

	//int numOfResult = 10;

	//for(int i = 0 ; i < numOfResult ; i++)
	//{
	//	tree<string>::iterator iter;
	//	iter = find(tr.begin(), tr.end() , to_string(leafErr[i]));
	//	iter--;

	//	vector<string> cfList = split_str((*iter), '_');
	//	
	//	Mat resultStack = userDraw.clone();
	//		
	//	for(vector<string>::iterator cf = cfList.begin() ; cf != cfList.end() ; cf++)
	//	{
	//		vector<string> cAndf = split_str((*cf), '*');
	//		int stackC = stoi(cAndf[0]);
	//		int stackF = stoi(cAndf[1]);
	//		//cout << stackC<<" "<<stackF<<endl;

	//		//get contour and food subsequence
	//		vector<Point> matchSeqDraw = subPointSeq(samplepointsOfDraw[sortedFragList[stackC].Element[stackF]["cIndex"]], sortedFragList[stackC].Element[stackF]["r"], sortedFragList[stackC].Element[stackF]["l"]);
	//		vector<Point> matchSeqFood = subPointSeq(samplepointsOfFood[sortedFragList[stackC].Element[stackF]["fIndex"]], sortedFragList[stackC].Element[stackF]["q"], sortedFragList[stackC].Element[stackF]["l"]);

	//		//get warping matrix
	//		Mat warpMat_2 = estimateRigidTransform(matchSeqFood, matchSeqDraw, false); //(src, dst)

	//		Mat resultStack_2 = resultStack.clone();
	//		warpAffine(imread(dir + files[sortedFragList[stackC].Element[stackF]["fIndex"] + 2], -1), resultStack_2, warpMat_2, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	//		//resultStackClone = resultStack.clone();
	//		resultStack = addTransparent(resultStack, resultStack_2);

	//	}
	//	imwrite("result_"+to_string(i)+".png", resultStack);


	//	//cout <<(*iter)<<endl;
	//}


	//singleTest();
	//testImg();
	//errTest();
	//recipeTest();
	topoTest();
	system("Pause");
}

//single test
void singleTest(void)
{


	//clock_t start = clock(); // compare start
	string tmp = "foodImg/040.png";
	string tmp2 = "foodImg/039.png";
	Mat input1 = imread(tmp, -1);
	Mat input2 = imread(tmp2, -1);

	descri descri1(tmp);
	Mat inputDes1 = descri1.resultDescri().clone();
	vector<Mat> inputDesSeq1 = descri1.seqDescri();
	descri descri2(tmp2);
	Mat inputDes2 = descri2.resultDescri();
	vector<Mat> inputDesSeq2 = descri2.seqDescri();

	for(int n = 0 ; n < inputDesSeq2.size() ; n++)
	{
		inputDesSeq2[n].at<double>((20-n+inputDesSeq2[n].rows)%inputDesSeq2[n].rows, (50-n+inputDesSeq2[n].cols)%inputDesSeq2[n].cols) = 255;
		//imwrite("desSeq/"+to_string(n)+"_.png", inputDesSeq2[n]);
	}

	imwrite("des1.png", inputDes1);
	imwrite("des2.png", inputDes2);

	clock_t start = clock(); // compare finish
	comp compDes(inputDesSeq1, inputDesSeq2, descri1.sampleResult(), descri2.sampleResult(), 0, 32+2, input1.size());
	//comp compDes(descri1.seqDescri(), descri2.seqDescri(), descri1.sampleResult(), descri2.sampleResult(), 0, 103+2, input1.size());
	clock_t finish = clock(); // compare finish
	cout << finish - start<<endl;
	fragList tmpppp;
	
	tmpppp.Element = compDes.fragList2();

	cout <<"fragSize: "<< tmpppp.Element.size()<<endl;;


	for(int i = 0 ; i < tmpppp.Element.size() ; i++)
		cout << "@start1: "<< tmpppp.Element[i].r <<" @start2: "<< tmpppp.Element[i].q <<" @range: "<< tmpppp.Element[i].l <<endl;

	Mat input1_draw = input1.clone();
	Mat input2_draw = input2.clone();

	Mat foodEdge = Mat::zeros(input1.size(), CV_8UC4);
	Mat foodEdgeAffine = Mat::zeros(input1.size(), CV_8UC4);
	Mat warpingResult = input1.clone();

	drawContours( foodEdge, vector<vector<Point> >(1, descri2.sampleResult()), 0 ,  Scalar(255,0,0,255), 1, 8);


	for(int i = 0 ; i < tmpppp.Element.size() ; i++)
	{
		//warpAffine(input2, warpingResult, tmpppp.Element[i].warpMatrix, warpingResult.size());
		//warpAffine(foodEdge, foodEdgeAffine, tmpppp.Element[i].warpMatrix, foodEdgeAffine.size());
		imwrite("testR/_"+to_string(i)+".png", addTransparent(input1, tmpppp.Element[i].warpResult));
		//imwrite("testR/_"+to_string(i)+"e.png", foodEdgeAffine);
	}

	//imwrite("result1.png", input1_draw);
	//imwrite("result2.png", input2_draw);
	//warpAffine(input2, warpingResult, warp_mat, warpingResult.size());
	//imwrite("warping.png", warpingResult);
}

//get test img
//void testImg(void)
//{
//	Mat userDraw = imread("inputImg/inin2.png", -1);
//
//	Mat cannyColor = cannyThreeCh(userDraw, true);
//
//	vector<vector<Point> > userDrawContours;
//	vector<Vec4i> hierarchy;
//
//	findContours(cannyColor.clone(), userDrawContours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, Point(0, 0) );
//	
//	vector<vector<Point> > disjointContour;
//
//	for(int i = 0 ; i < userDrawContours.size() ; i++)
//	{
//		if(/*userDrawContours[i].size()>70 && */hierarchy[i][3] != -1)
//			disjointContour.push_back(userDrawContours[i]);
//	}
//
//	sort(disjointContour.begin(), disjointContour.end(), compareContourSize);
//
//	descri descri1(disjointContour[0]);
//	vector<Mat> des1 = descri1.seqDescri();
//	vector<Point> seq1 = descri1.sampleResult();
//
//	string tmp2 = "foodImg/086.png";
//	Mat input2 = imread(tmp2, -1);
//	descri descri2(tmp2);
//	vector<Mat> des2 = descri2.seqDescri();
//	vector<Point> seq2 = descri2.sampleResult();
//
//	cout << seq1.size() <<" "<<seq2.size()<<endl;
//	cout << "!!!"<<endl;
//	comp com(des1, des2, seq1, seq2, 0, 0);
//
//	fragList fragL;
//	fragL.Element= com.fragList2();
//
//	cout << fragL.Element.size();
//
//	Mat warpingResult = userDraw.clone();
//	for(int i = 0 ; i < fragL.Element.size() ; i++)
//	{
//		warpAffine(input2, warpingResult, fragL.Element[i].warpMatrix, warpingResult.size());
//		imwrite("testR/_8"+to_string(i)+".png", warpingResult);
//		//imwrite("testR/_"+to_string(i)+"e.png", foodEdgeAffine);
//	}
//
//}

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

//err test
void errTest(void)
{
	string path1 = "errorTest/inin.png";
	string path2 = "errorTest/_0.png";
	Mat origin = imread("inputImg/duck01.png", -1);
	Mat warp0 = imread("621/0.310566.png", -1);
	Mat warp1 = imread("621/0.313224.png", -1);
	Mat warp2 = imread("621/0.314438.png", -1);
	Mat warp3 = imread("621/0.316784.png", -1);

	//Mat fail = imread("errorTest/erre.png", -1);
	//cout << edgeError(origin, warp0)<<endl;
	//cout << edgeError(origin, warp1)<<endl;
	//cout << edgeError(origin, warp2)<<endl;
	//cout << edgeError(origin, warp3)<<endl;
	//cout << edgeError(origin, fail)<<endl;

	cout << interError(origin, warp0)<<endl;
	cout << interError(origin, warp1)<<endl;
	cout << interError(origin, warp2)<<endl;
	cout << interError(origin, warp3)<<endl;

	//Mat origin2 = imread("errorTest/inin2.png", -1);
	//Mat ref0 = imread("errorTest/_0.png", -1);
	//Mat ref1 = imread("errorTest/_0_2.png", -1);
	//Mat ref2 = imread("errorTest/_0_3.png", -1);
	//int tmp;
	//cout <<"reference error: "<< refError(origin, warp0, tmp)<<endl;
	//cout <<"next Index: "<< tmp <<endl;
	//cout <<"reference error: "<< refError(origin, warp1, tmp)<<endl;
	//cout << "next Index: "<<tmp <<endl;
	//cout <<"reference error: "<< refError(origin, warp2, tmp)<<endl;
	//cout << "next Index: "<<tmp <<endl;
	//cout <<"reference error: "<< refError(origin, warp3, tmp)<<endl;
	//cout << "next Index: "<<tmp <<endl;

	//Mat src = imread("inputImg/inin2.png", -1);
	//Mat color1 = imread("testR/_0.png", -1);
	//Mat color2 = imread("testR/_80.png", -1);
	//Mat color3 = imread("testR/_e0.png", -1);

	//cout << "color error1: "<< colorError(origin, warp0)<<endl;
	//cout << "color error2: "<< colorError(origin, warp1)<<endl;
	//cout << "color error3: "<< colorError(origin, warp2)<<endl;
	//cout << "color error3: "<< colorError(origin, warp3)<<endl;
}

//recipe test
void recipeTest(void)
{
	recipe rec1;
	rec1.addFood(92, 20, "plum", false);
	rec1.addFood(146, 20, "wax", false);
	rec1.addFood(79, 1, "pineapple", false);
	rec1.addFood(8, 1, "donut", false);
	rec1.addFood(38, 20, "cucumber", false);
	rec1.addFood(41, 20, "oliver", false);
	rec1.addFood(51, 20, "cherry", false);
	rec1.addFood(13, 20, "orange", false);
	
	rec1.writeRecipeFile("flower.bin");

	recipe rec2;
	rec2.readRecipeFile("flower.bin");
	//rec2.reduceFood(3);
	rec2.print();
	rec2.reset();
	rec2.print();
}

//topo test
void topoTest(void)
{
	topo G(4);
	
	G.addEdge(0, 1);
	G.addEdge(1, 2);
	G.addEdge(2, 3);
	G.addEdge(2, 0);
	G.printAdjList();

	if(G.isCyclic())
        cout << "Graph contains cycle";
    else
        cout << "Graph doesn't contain cycle";
    
	cout <<endl;
	G.delEdge(2, 3);
	G.printAdjList();

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
Mat cannyThreeCh(Mat input, bool mode)
{
	vector<Mat> channels;
	
	split(input, channels);

	Mat B = channels[0];
	Mat G = channels[1];
	Mat R = channels[2];
	Mat A = channels[3];


	for(int i = 0 ; i < input.rows ; i++)
	{
		for(int j = 0 ; j < input.cols ; j++)
		{
			if(A.at<uchar>(i, j) == 0)
			{
				B.at<uchar>(i, j) = 0;
				G.at<uchar>(i, j) = 0;
				R.at<uchar>(i, j) = 0;

			}
		}
	}

	int lowTh;
	int highTh;

	if(mode)
	{
		lowTh = 50;
		highTh = 150;
	}
	else
	{
		lowTh = 250;
		highTh = 750;
	}

	Mat cannyB, cannyG, cannyR, cannyA;

	Canny(B, cannyB, lowTh, highTh, 3);
	Canny(G, cannyG, lowTh, highTh, 3);
	Canny(R, cannyR, lowTh, highTh, 3);
	Canny(A, cannyA, lowTh, highTh, 3);
	Mat cannyColor;

	bitwise_or(cannyB, cannyG, cannyColor);
	bitwise_or(cannyColor, cannyR, cannyColor);
	bitwise_or(cannyColor, cannyA, cannyColor);

	return cannyColor;
}

//canny alpha channel
Mat cannyAlpha(Mat input)
{
	vector<Mat> channels;
	split(input, channels);

	Mat A = channels[3];

	Mat cannyA;

	Canny(A, cannyA, 50, 150, 3);

	return cannyA;
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
	Mat drawEdge = cannyThreeCh(draw, true);
	Mat foodEdge = cannyThreeCh(food, true);

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
			/*
			if(drawAlphaBin.at<int>(i, j) == 1 && foodAlphaBin.at<int>(i, j) == 1)
			{
				Vec4b pixDraw = draw.at<Vec4b>(i, j);
				Vec4b pixFood = food.at<Vec4b>(i, j);

				tmp =  sqrt(pow(pixDraw.val[0]-pixFood.val[0], 2) + pow(pixDraw.val[1]-pixFood.val[1], 2) + pow(pixDraw.val[2]-pixFood.val[2], 2));
				score += tmp;
			}
			else if(drawAlphaBin.at<int>(i, j) == 1 && foodAlphaBin.at<int>(i, j) == 0)
			{
				tmp = sqrt(3*(255*255));
				score += tmp;
			}
			*/
			if(orValue == 1)
			{
				Vec4b pixDraw = draw.at<Vec4b>(i, j);
				Vec4b pixFood = food.at<Vec4b>(i, j);

				tmp =  sqrt(pow(pixDraw.val[0]-pixFood.val[0], 2) + pow(pixDraw.val[1]-pixFood.val[1], 2) + pow(pixDraw.val[2]-pixFood.val[2], 2));
				score += tmp;

			}
		}
	}

	return score/(draw.rows*draw.cols);
}

//reference error
double refError(Mat draw, Mat food, int& nextIndex)
{
	Mat drawEdge = cannyThreeCh(draw, true);
	Mat foodEdge = cannyThreeCh(food, true);

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
			if(pointDist.size() > 0)
			{
				double tmp = *min_element(pointDist.begin(), pointDist.end());
				//cout << tmp << endl;
				contourErr += tmp;
				score += tmp;
				pointDist.clear();
			}
		}
		perContourErr.push_back(contourErr/disjointContour[i].size());
		contourErr = 0;
	}

	/*for(int i = 0 ; i < perContourErr.size() ; i++)
		cout << "contour "<< i <<": "<< perContourErr[i]<<endl;*/

	nextIndex = distance(perContourErr.begin(), max_element (perContourErr.begin(), perContourErr.end()));

	return score/drawPointNum;
}

//intersection error
double interError(Mat draw, Mat food)
{
	Mat drawAlphaBin = alphaBinary(draw);
	Mat foodAlphaBin = alphaBinary(food);
	int intersectionArea = 0;
	int unionArea = 0;

	for(int i = 0 ; i < foodAlphaBin.cols ; i++)
	{
		for(int j = 0 ; j < foodAlphaBin.rows ; j++)
		{
			if(drawAlphaBin.at<int>(i, j) == 1 || foodAlphaBin.at<int>(i, j) == 1)
			{
				unionArea++;
				if(drawAlphaBin.at<int>(i, j) == 1 && foodAlphaBin.at<int>(i, j) == 1)
					intersectionArea++;
			}
		}
	}

	cout << intersectionArea<<" "<<unionArea<<endl;
	double ratio = (double)intersectionArea/(double)unionArea;
	return (1 - ratio);
}

//gradient error 
double gradError(Mat draw, Mat food)
{
	Mat blurDraw;
	Mat blurFood;

	GaussianBlur(draw, blurDraw, Size(3, 3), 0, 0);
	GaussianBlur(food, blurFood, Size(3, 3), 0, 0);

	Mat gradDraw = cannyThreeCh(blurDraw, true);
	Mat gradFood = cannyThreeCh(blurFood, true);

	Mat subResult;
	absdiff(gradDraw, gradFood, subResult);

	return sum(subResult).val[0]/(subResult.cols*subResult.rows);
}

// iError with two contour
double iContourError(vector<Point> contour1, vector<Point> contour2, Mat food)
{
	Mat drawGray, foodGray;
	//cvtColor(food, foodGray, CV_BGR2GRAY);

	Mat drawNZ, foodNZ;
	findNonZero(food, foodNZ);
	
	double minDist = 0.0;
	double totalDist = 0.0;

	for(int c = 0 ; c < contour1.size() ; c++)
	{
		for(int i = 0 ; i < foodNZ.rows ; i++)
		{
			Point locF = foodNZ.at<Point>(i);
			double dist = sqrt(pow(locF.x-contour1[c].x, 2) + pow(locF.y-contour1[c].y, 2));
			if(i == 0)
				minDist = dist;
			else
				if(minDist>dist)
					minDist = dist;
		}
		totalDist += minDist;
	}

	for(int c = 0 ; c < contour2.size() ; c++)
	{
		for(int i = 0 ; i < foodNZ.rows ; i++)
		{
			Point locF = foodNZ.at<Point>(i);
			double dist = sqrt(pow(locF.x-contour2[c].x, 2) + pow(locF.y-contour2[c].y, 2));
			if(i == 0)
				minDist = dist;
			else
				if(minDist>dist)
					minDist = dist;
		}
		totalDist += minDist;
	}

	double totalPoints =(double) (contour1.size()+contour2.size());

	return totalDist/totalPoints;
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
double getTotalErr(int state, int& nextIndex, int& nextFrag, double& refErr, vector<vector<Point> >& samplepointsOfDraw, vector<vector<Point> >& samplepointsOfFood, vector<fragList>& sortedFragList, Mat& resultStack, Mat& resultStackClone, Mat& stackEdge, Mat& stackEdgeClone, string& dir, Mat& userDraw) {

	//Mat tmpContour = Mat::zeros(userDraw.size(), CV_8UC4);
	//drawContours(tmpContour, samplepointsOfFood, 0, Scalar(0, 0, 255, 255), 1, 8);
	//if (state == 0) {
	//	resultStack = userDraw.clone();
	//	warpAffine(tmpContour, stackEdge, sortedFragList[nextIndex].Element[nextFrag].warpMatrix, stackEdge.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	//	warpAffine(imread(dir + files[sortedFragList[nextIndex].Element[nextFrag].fIndex + 2], -1), resultStack, sortedFragList[nextIndex].Element[nextFrag].warpMatrix, resultStack.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	//}
	//else {
	//	Mat resultStack_2 = resultStack.clone();
	//	Mat stackEdge_2 = stackEdge.clone();
	//	warpAffine(tmpContour, stackEdge_2, sortedFragList[nextIndex].Element[nextFrag].warpMatrix, stackEdge_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	//	warpAffine(imread(dir + files[sortedFragList[nextIndex].Element[nextFrag].fIndex + 2], -1), resultStack_2, sortedFragList[nextIndex].Element[nextFrag].warpMatrix, resultStack_2.size(), CV_INTER_LINEAR, cv::BORDER_CONSTANT);
	//	resultStackClone = resultStack.clone();
	//	stackEdgeClone = stackEdge.clone();
	//	bitwise_or(stackEdge_2, stackEdge, stackEdge);
	//	resultStack = addTransparent(resultStack, resultStack_2);
	//}
	refErr = refError(userDraw, resultStack, nextIndex);
	double totalErr = edgeError(userDraw, stackEdge) + colorError(userDraw, resultStack) + refErr;
	return totalErr;
}

//void vecSeqIndex(fragList &input)
//{
//	//fragList tmp;
//	//tmp.Element.assign(input.Element.begin(), input.Element.end());
//	//fragList copyTmp(input.begin(), input.end()); 
//	sort(tmp.Element.begin(), tmp.Element.end(), compareWithCertainKey);
//
//	//return tmp;
//}

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

	vector<int> getSize;
	ifstream fin("preData/foodDesSize.bin", ios::in | ios::binary);
	int num;
	fin.read((char *)&num, sizeof(num));
	for (int i = 0; i<num; ++i) {
		int buff;
		fin.read((char *)&buff, sizeof(buff));
		getSize.push_back(buff);
	}

	vector<Mat> get_foodDesSeq = vecmatread("preData/foodDesSeq.bin");
	vector<Point> get_sampleResult = vecPointRead("preData/sampleResult.bin");
	vector<Mat> buff_foodDesSeq;
	vector<Point> buff_sampleResult;

	for (int i = 0; i < get_foodDesSeq.size(); i++)	{
		buff_foodDesSeq.push_back(get_foodDesSeq[i]);
		buff_sampleResult.push_back(get_sampleResult[i]);
		count++;
		if (count == getSize[idx]) {
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

