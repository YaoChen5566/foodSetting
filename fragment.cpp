#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>

#include "fragment.h"

frag::frag()
{
	
}

void frag::setInfo(int _r, int _q, int _l, double _score, int _cIndex, int _fIndex, Mat _warpMatrix, Mat _warpResult)
{
	r = _r;
	q = _q;
	l = _l;
	fIndex = _fIndex;
	cIndex = _cIndex;
	score = _score;
	warpMatrix = _warpMatrix;
	warpResult = _warpResult;
}

void frag::setError(double _eError, double _cError, double _rError, double _iError, double _iErrorRatio1)
{
	eError = _eError;
	cError = _cError;
	rError = _rError;
	iError = _iError;
	iErrorRatio1 = _iErrorRatio1;
	sError = eError+cError+rError+iError;
}

bool frag::theSame(frag A)
{
	if(q == A.q && r == A.r)
		return true;
	else if(abs(q-r) == abs(A.q-A.r))
		return true;
	else
		return false;
}