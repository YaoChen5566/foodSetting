#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>

#include "fragment.h"

frag::frag()
{
	
}

void frag::setInfo(int _r, int _q, int _l, int _fIndex, int _cIndex, double _score, Mat _warpMatrix)
{
	r = _r;
	q = _q;
	l = _l;
	fIndex = _fIndex;
	cIndex = _cIndex;
	score = _score;
	warpMatrix = _warpMatrix;
}

void frag::setError(double _eError, double _cError, double _rError, double _iError)
{
	eError = _eError;
	cError = _cError;
	rError = _rError;
	iError = _iError;
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