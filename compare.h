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
#include "fragment.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

class comp
{
public:

	comp(Mat des1, vector<Mat> des2Seq);
	comp();
	comp(vector<Mat> des1, vector<Mat> des2Seq, vector<Point> pointSeq1, vector<Point> pointSeq2, int contourIndex, int foodIndex, Size drawSize);
	comp(Mat foodImg, Mat mapRQ, vector<Point> pointSeq1, vector<Point> pointSeq2);

	//frag fragment(int rS, int qS, int mL);

	vector<frag> fragList2();
	vector<frag> fragList3();

	vector<map<string, int> > fragList();

	int startIndex1();
	int startIndex2();
	int range();
	double score();
	//int n;

	
	
private:
	void setInitial();
	void setScoreThreshold();
	void compareDes(Mat input1, Mat input2);
	void compareDesN(Mat input1, Mat input2, int index, bool cLarge);
	void compareDesN2(Mat input1, Mat input2, int index);
	void clearFrag();
	void disFrag();
	void localMin();
	Mat subMatrix(Mat input, int row, int col, int range);
	Mat normalizeRQ();
	void localMaxOfRQMap();
	vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int matchL);
	void get_Min_Max(Mat &rqmap, int windowSize);
	double occupiedInterval();
	void findCombination();
	


	int _startIndex1;
	int _startIndex2;
	int _range;
	int _fIndex;
	int _cIndex;
	double _thresholdScore;
	double _score;
	double _minScore;
	int _rDesSize;
	int _qDesSize;
	double _ratio1;
	
	vector<Point> _maxVec;
	vector<Point> _minVec;
	Mat _mapRQ; //RQmap with match length
	Mat _mapScore; //RQmap with score
	Mat _warpResult; // food after warping
	Size _drawSize; //userDraw image size
	vector<vector<Mat> > _warpMatrixMap; //RQmap with warpmatrix
	vector<frag> _frag2;
	vector<frag> _frag3;
	vector<map<string, int> > _frag;
	vector<map<string, int> > _totalFrag;
	vector< map<string, int> > _clearResult;
	vector<Point> _pointSeq1;
	vector<Point> _pointSeq2;

	double imageOverlap(vector<Point> newPointSeq);
	bool fragExist(map<string, int> newFrag);
	bool fragSame(map<string, int> frag1, map<string, int> frag2);
};