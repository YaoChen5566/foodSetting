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
//#include "fragment.h"

# define PI 3.1415926

using namespace std;
using namespace cv;

class comp
{
public:

	comp(Mat des1, vector<Mat> des2Seq);
	comp();
	comp(Mat des1, Mat des2);
	comp(Mat des1, vector<Mat> des2Seq, vector<Point> pointSeq1, vector<Point> pointSeq2, int contourIndex, int foodIndex);

	//frag fragment(int rS, int qS, int mL);

	//vector<frag> fragList();
	vector<map<string, int>> fragList();

	int startIndex1();
	int startIndex2();
	int range();
	double score();
	//int n;

	
	
private:
	void setInitial();
	void setScoreThreshold();
	void compareDes(Mat input1, Mat input2);
	void compareDesN(Mat input1, Mat input2, int index);
	Mat subMatrix(Mat input, int row, int col, int range);
	vector<Point> subPointSeq(vector<Point> inputSeq, int startIndex, int matchL);


	int _startIndex1;
	int _startIndex2;
	int _range;
	int _fIndex;
	int _cIndex;
	double _thresholdScore;
	double _score;
	//vector<frag> _frag;
	vector<map<string, int>> _frag;
	vector<Point> _pointSeq1;
	vector<Point> _pointSeq2;

	bool fragExist(map<string, int> newFrag);
	bool fragSame(map<string, int> frag1, map<string, int> frag2);
};