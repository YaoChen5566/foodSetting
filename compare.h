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

class comp
{
public:
	
	int startIndex1;
	int startIndex2;
	int range;
	double score;
	int n;

	comp(Mat des1, Mat des2);
	comp(Mat des1, vector<Mat> des2Seq);
	
private:
	void setScoreThreshold();
	void compareDes(Mat input1, Mat input2);
	void compareDes(Mat input1, Mat input2, int index);
	Mat subMatrix(Mat input, int row, int col, int range);

};