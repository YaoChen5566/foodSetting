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

	frag(int _r, int _q, int _l);

	int r; // reference start
	int q; // query start
	int l; // match length

	
	bool theSame(frag A);
private:

	void setInfo(int rS, int qS, int mL);

};

#endif