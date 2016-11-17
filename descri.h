#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

# define PI 3.1415926

using namespace std;
using namespace cv;

class descri
{
public:
	
	descri(string &imgPath);
	descri(vector<Point> contour);
	
	vector<Point> sampleResult();

	Mat resultDescri();
	vector<Mat> seqDescri();
	//Mat grayDescri;

private:
	void imgToDes(Mat input);
	Mat alphaBinary(Mat input);
	int maxContour(vector<vector<Point>> contours);
	float contourLength(vector<Point> singleContour);
	void getSamplePoints(vector<Point> singleContour);
	void descriptor(vector<Point> samplePoints);
	void getSeqDescriptor(vector<Point> samplePoints);
	//void desToGrayImg(Mat input);
	float angle(Point i, Point j, Point jMinusDelta);
	
	vector<Point> _sampleResult;
	Mat _resultDescri;
	vector<Mat> _seqDescri;

};