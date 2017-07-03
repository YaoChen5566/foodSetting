#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>

#ifndef fragment_H
#define fragment_H

using namespace std;
using namespace cv;

class frag
{
public:

	frag();

	int r; // reference start
	int q; // query start
	int l; // match length
	int fIndex; //file index
	int cIndex; //contour index
	double score; //fragment score
	//Mat warpMatrix; //warping matrix
	Mat warpResult; //food after warping
	vector <Mat> warpMatSeq; // warpMat seq
	double eError; //edge error
	double cError; //color error
	double rError; //reference error
	double sError; //sum of three error
	double iError; //intersection error
	double iErrorRatio1; // intersection error for contour

	void setInfo(int _r, int _q, int _l, double _score, int _cIndex, int _fIndex, vector<Mat> _warpMatrixSeq, Mat warpResult);
	void setError(double _eError, double _cError, double _rError, double _iError, double _iErrorRatio1);
	bool theSame(frag A);
private:


};

#endif