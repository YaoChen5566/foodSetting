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
	Mat warpMatrix; //warping matrix
	double eError; //edge error
	double cError; //color error
	double rError; //reference error
	double sError; //sum of three error
	double iError; //intersection error

	void setInfo(int _r, int _q, int _l, int _fIndex, int _cIndex, double _score, Mat _warpMatrix);
	void setError(double _eError, double _cError, double _rError, double _iError);
	bool theSame(frag A);
private:


};

#endif