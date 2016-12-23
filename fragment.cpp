#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>
#include <iostream>

#include "fragment.h"

frag::frag(int _r, int _q, int _l)
{
	setInfo(_r, _q, _l);
}

void frag::setInfo(int rS, int qS, int mL)
{
	r = rS;
	q = qS;
	l = mL;
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