
//////////////////////////////////////////////////////////////////////
//. Disclaimer:
// ************************************************************
// Warning! This contains the **WHOLE** workspace over the development (at least most of it) including most of the attempts
// and is NOT the final version that was submitted (may be the LATEST WORKING REVISION but not the version which was cleaned up and submitted but the one I kept for future reference, personal use and records) but because of being in its early stages
// of dev, contains the initial RAW version of ALL (most) of the code present. This means that only the functions USED in the code are the one's relevant.
// ALSO USE AT YOUR OWN RISK.
// ************************************************************
// ************************************************************

/*


Latest Working Revision

Motion Detection for surveying the ganges river dolphin
::Gaurav Dass

Done under the guidance of::::

Prof. Dr. Arun Kumar
Prof. Dr. Rajender Bahl

*/




#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <cstdarg>
#include "opencv2/opencv.hpp"
#include "fstream"
#include "Kalman.h"
#include "HungarianAlg.h"
#include <dirent.h>
#include <math.h>
#include <time.h>
#include <opencv2/videostab.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "new_stabilize.h"
#include <memory>

#define NORM_CPP(x) sqrt(pow(x,2))

using namespace std;
using namespace cv;
using	namespace	cv::videostab;

void homomor(Mat src, Mat &dst,Mat kernel)
{
// dst = Mat(src.size(),src.type());
int srctype = src.type();
	
src.convertTo(src,CV_32F);
vector<Mat>srcC(3);
split(src,srcC);

for(int k = 0;k<3;k++)
{
Mat temp;
int sigmaC = 200;
temp = srcC[k];
add(temp,1,temp);
log(temp,temp);
filter2D(temp,temp,-1,kernel);
exp(temp,temp);			  // Reduce Variables
temp = temp - 1;
temp.copyTo(srcC[k]);
}
	
merge(srcC,dst);  vector<Mat>().swap(srcC);	
dst.convertTo(dst,srctype);
	
}

void homomor2(Mat src, Mat &dst,int rad)
{
// dst = Mat(src.size(),src.type());
int srctype = src.type();
	
src.convertTo(src,CV_64F);
Size IO = src.size(); Point2f G;
G.x = IO.width/2;
G.y = IO.height/2;

vector<Mat>srcC(3);
split(src,srcC);

for(int k = 0;k<3;k++)
{
Mat temp;
int sigmaC = 200;
temp = srcC[k];
add(temp,1,temp);
log(temp,temp);
		
		
int M = 2*(src.rows) + 1;
int N = 2*(src.cols) + 1;	

Size dftsize;
dftsize.width = N;
dftsize.height = M;

Mat H = Mat(M,N,CV_64F);
// Mat FF = create_filter(H, 7, 3, 1, 0); // create gaussian filter				
Mat FF = create_filter_twolayer(H, rad , 3, 1, 0); // create gaussian filter
cvShiftDFT(FF,FF);
// shift(FF,FF,G);
Mat Hema = 1.5*FF + 0.5; FF.release();

Mat If,Iout;
Mat tmpA = Mat( dftsize,CV_64F, Scalar(0) );
Mat tmpIf = Mat( dftsize,CV_64F, Scalar(0) );

Mat roiA = Mat(tmpA,Rect(0,0,src.cols,src.rows));
//Mat roiIf = Mat(tmpIf,Rect(0,0,Iout.cols,Iout.rows));
temp.copyTo(roiA);
//If.copyTo(roiIf);
dft( tmpA,tmpIf,DFT_COMPLEX_OUTPUT,0);   // Real output necessary only????
			
vector<Mat> spl_If;
vector<Mat> Hm;
split(tmpIf,spl_If);
split(Hema,Hm);
multiply(spl_If[0],Hm[0],spl_If[0]);
multiply(spl_If[1],Hm[1],spl_If[1]);
merge(spl_If,tmpIf); Hema.release();  vector<Mat>().swap(Hm);  vector<Mat>().swap(spl_If);
// mulSpectrums(tmpIf,Hema,tmpIf,DFT_ROWS);
dft( tmpIf,Iout, DFT_INVERSE + DFT_SCALE + DFT_REAL_OUTPUT,0); tmpIf.release(); // test these;
Mat Inew = Iout(Rect(0,0,src.cols,src.rows));
// Inew = Iout(Rect(0,0,src.cols,src.rows)); Iout.release();
exp(Inew,Inew); Iout.release();
Inew = Inew - 1;
srcC[k] = Inew;
	}
	
merge(srcC,dst);  vector<Mat>().swap(srcC);	
dst.convertTo(dst,srctype);
}

Mat getreducemat(Mat mgpts,Mat scpts,Mat a13,Mat a46)
{

	Mat red,A13,A46,newmg,newsc,XX,YY;
	
	Mat SCP[] = {scpts,Mat::ones(scpts.rows,1,scpts.type()) };
	hconcat(SCP,2,newsc);
	
	repeat(a13, 1,scpts.rows, A13);	A13 = A13.t();
	repeat(a46,1,scpts.rows, A46);	A46 = A46.t();
	
	multiply(A13,newsc,A13);
	multiply(A46,newsc,A46);
	
	reduce(A13,XX,1,CV_REDUCE_SUM);
	reduce(A46,YY,1,CV_REDUCE_SUM);

	Mat rree[] = {XX,YY};
	hconcat(rree,2,red);
	
	red = mgpts - red;
	
	return red;
	
}

Mat get_affine_matrix(Mat movpoints,Mat srcpoints)			// srcpoints = moving frame ||| movpoints = static frame
// Mat get_affine_matrix(Mat srcpoints,Mat movpoints)			// srcpoints = moving frame ||| movpoints = static frame
{
float epsilon = 0.01;
Mat ontm = Mat::ones(srcpoints.rows, 1, srcpoints.type());
Mat A,X,Y,a13, a46;
	
Mat mtarr[] = {srcpoints, ontm};
hconcat(mtarr, 2 , A);

movpoints.col(0).copyTo(X);
movpoints.col(1).copyTo(Y);

/// FOR TESTING PURPOSES ONLY REMOVE LATER	

 a13 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * X);		// Check for SVD parameters for accurate warp
 a46 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * Y);

// a13 = (A.inv(DECOMP_SVD)) * X;
// a46 = (A.inv(DECOMP_SVD)) * Y;
float J = 0,Jold = 0,delJ = 0,olddelJ = 0;
float thep = 100;
int cco = 0;	
	Mat Xresidue, Yresidue,XW,YW;

	// loop condition to be put here
	do{

	Mat redu =  getreducemat(movpoints,srcpoints,a13,a46);
	Mat W;	
	divide(1,abs(redu) + epsilon,W);
	
	multiply(A.col(0),W.col(0),A.col(0));
	multiply(A.col(1),W.col(1),A.col(1));
	
	multiply(X,W.col(0),X);
	multiply(Y,W.col(1),Y);
	
 	a13 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * X);		// Check for SVD parameters for accurate warp
 	a46 = (A.t() * A).inv(DECOMP_CHOLESKY) * (A.t() * Y);	
	
	pow(redu,2,redu);
		
	Jold = J;		
	J = sum(redu.col(0))[0] + sum(redu.col(1))[0];
	olddelJ = delJ;
	delJ = abs(J - Jold);
		
	cco++;	
	}while(	(delJ > thep) && (cco > 0) & (( (abs(delJ - olddelJ) == 0)) || (abs(delJ - olddelJ) <= 1.5)) ) ;

Mat affine_matrix; Mat tmpone  = Mat::zeros(1,3,CV_32F); tmpone.at<float>(2) = 1;
affine_matrix.push_back(a13.t());
affine_matrix.push_back(a46.t());
//affine_matrix.push_back(tmpone);

	
return affine_matrix;
}

Mat register_fram(Mat refframe, Mat movingframe, vector<Point2f> &srcPf, int cnt )			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	
	if( cnt%20 == 0 )
	{
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	}
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
	
	Mat warp_m;
	
	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 )
	{
	warp_m = Mat::eye(2,3,CV_32F);
	}
	else
	{
	warp_m = get_affine_matrix(mvgpts,srcpts);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpAffine(movingframe,warped_frame,warp_m, movingframe.size());
	
	srcPf = dstPf;
	return warped_frame;

}

Mat register_fram_dynamic(Mat refframe, Mat movingframe)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	if(!srcPf.size()){return movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return movingframe;
	}
	else
	{
	warp_m = get_affine_matrix(mvgpts,srcpts);
	}
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
	
	warpAffine(movingframe,warped_frame,warp_m, movingframe.size());
	return warped_frame;

}

Mat horizon_detect(Mat img)
{
Mat element = getStructuringElement( MORPH_ELLIPSE, Size(2,2) );
	Mat diamond = Mat(5,5,CV_8U,Scalar(1));
            diamond.at<uchar>(0,0)= 0; 
            diamond.at<uchar>(0,1)= 0;
            diamond.at<uchar>(1,0)= 0;
            diamond.at<uchar>(4,4)= 0;
            diamond.at<uchar>(3,4)= 0;
            diamond.at<uchar>(4,3)= 0;
            diamond.at<uchar>(4,0)= 0;
            diamond.at<uchar>(4,1)= 0;
            diamond.at<uchar>(3,0)= 0;
            diamond.at<uchar>(0,4)= 0;
            diamond.at<uchar>(0,3)= 0;
            diamond.at<uchar>(1,4)= 0;

vector<Mat> ima(3);
Mat imgopen, imgclos;
morphologyEx(img,imgopen,MORPH_OPEN,element);
morphologyEx(imgopen,imgclos,MORPH_CLOSE,element);

Mat temefr1,temefr2,tframe;
dilate( imgclos,temefr1, diamond);
morphologyEx( imgclos,temefr2, MORPH_CLOSE, diamond);

tframe = temefr1 - temefr2;
dilate(tframe,tframe,diamond);
dilate(tframe,tframe,diamond);
dilate(tframe,tframe,diamond);
dilate(tframe,tframe,diamond);
dilate(tframe,tframe,diamond);


	
Mat t111; 
Mat tcopy; tframe.copyTo(tcopy);	
cvtColor(tframe,t111,CV_BGR2GRAY );
Mat BW;
int ots = threshold(t111,BW,255,1,THRESH_OTSU);
	
vector<Vec4i> lines;
Vec4i linnn;
///
// Vector indice generator


Mat nnne;

	Mat binlab,binlab8;
	int nolab8 = connectedComponents(BW,binlab8,8);
	Mat labres = Mat::zeros(img.size(),CV_8UC1);
	
	for(int i = 0; i< nolab8; i++)
	{
	vector<Point> pnts;

		for(int r = 0; r < img.rows; r++)
		{	for( int c = 0; c< img.cols; c++ )
			{

	if(binlab8.at<int>(r,c) == i)
	{
	Point pnt;
	pnt.x = c;
	pnt.y = r;
	pnts.push_back(pnt);		
	}
		}
			}

	
	Rect BR = boundingRect(Mat(pnts));

		if( (BR.width >= 0.99* img.cols) && (BR.height >= 0.99* img.rows)  )
		{
		continue;				
		}
		
rectangle(labres, BR.tl(),BR.br(),Scalar(255), -1,8,0);


//	labres(BR) = 255;

	}
// horizon line detector
//////

dilate(labres,labres,Mat());
dilate(labres,BW,Mat());

	for(int i = 0; i< BW.rows;i++ )
	{
		for(int j = 0; j< BW.cols;j++)
		{

			if( BW.at<uchar>(i,j) != 0 )
			{
		Mat nn = Mat(1,1,CV_32SC2);
		nn.at<Vec2i>(0)[0] = j;
		nn.at<Vec2i>(0)[1] = i;
		nnne.push_back(nn);				

			}
		}

	}

//
	
vector <Vec4i> hlines;
HoughLinesP(BW*255,hlines,1, CV_PI/180,10,300,0);

	
	Mat hpli = Mat::ones(img.size(), CV_8UC1) * 255;
	Vec4i maxline;
	double max = 0;
	int maxiter = 0;
	for(int i = 0; i < hlines.size(); i++ )		// Maximum Y extractor
	{
		Point pt1, pt2;
		Vec4i llo = hlines[i];

		if( (llo[1] > hpli.rows / 2) || (llo[3] > hpli.rows / 2) )		// To remove outliers which possibly cannot be the horizon
		{
		continue;
		}
		
		if(llo[1] > max)
		{
		max = llo[1];
		}
		
		if(llo[3] > max)
		{
		max = llo[3];
		}
		
		
		
	}
	
	hpli(Rect(0,0,hpli.cols,max)).setTo(0); // Inverting mask to make image region positives (255) and horizon negatives (0), bitwise_not on a 1 keeps is treated as a zero hence init value 255
	
//	maxline = hlines[maxiter];
//	line(hpli,Point(maxline[0],maxline[1]),Point(maxline[2],maxline[3]),Scalar(255),1,8,0);
	
	split(img,ima);
	bitwise_and(ima[0],hpli,ima[0]);
	bitwise_and(ima[1],hpli,ima[1]);
	bitwise_and(ima[2],hpli,ima[2]);
	merge(ima,img);
	
return img;

}

void stab_video(string inputPath)
{
string filnam;
string temname(inputPath);


	
size_t brck = inputPath.find_last_of("/");
size_t pnt = inputPath.find_last_of(".");

temname.replace(brck+1,pnt-1,"temp");
// filnam = inputPath.substr(brck+1,pnt - brck - 1);
					
	
int result  = rename(inputPath.data(),temname.data());
if(result != 0) {cout<<"\nRENAMING FILE FAILED\n"; exit(1); }

VideoCapture cap;
cap.open(temname.data());
	
VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open(inputPath.data(),codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));
	
Mat fr;
cap>>fr;		wir<<fr;
	
int i = 1;

	while (i < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
	
	Mat fra1;
	cap>>fra1;
	Mat ttmp = register_fram_dynamic(fr,fra1);
	wir<<ttmp;
	fra1.copyTo(fr);
	i++;
		
	}

	cout<<"\n InputPath Name:"<<inputPath.data()<<"\n";

cap.release();        
remove(temname.data());
}


Mat get_pitch_warp(Mat movpoints,Mat srcpoints)			// srcpoints = moving frame ||| movpoints = static frame
{

Mat R = Mat::eye(3,3,CV_32F);
float movX,movY,srcX,srcY;
movX = sum(movpoints.col(0))[0] / movpoints.rows;
movY = sum(movpoints.col(1))[0] / movpoints.rows;

srcX = sum(srcpoints.col(0))[0] / srcpoints.rows;
srcY = sum(srcpoints.col(1))[0] / srcpoints.rows;	
	
Mat T1 = Mat::eye(3,3,CV_32F);	
Mat T2 = Mat::eye(3,3,CV_32F);	

T1.at<float>(0,2) = -movX;	
T1.at<float>(1,2) = -movY;	
	
T2.at<float>(0,2) = -srcX;	
T2.at<float>(1,2) = -srcY;	

// movpoints = T2 * movpoints.t();	
// srcpoints = T1 * srcpoints.t();

	movpoints = movpoints.t();
	movpoints.push_back( Mat::ones(1,movpoints.cols,CV_32F) );
	movpoints = movpoints.t();
	
	srcpoints = srcpoints.t();
	srcpoints.push_back( Mat::ones(1,srcpoints.cols,CV_32F) );
	srcpoints = srcpoints.t();
	
	
for(int i = 0; i< movpoints.rows; i++)			// moving the coordinate systems to their common centers
{
	
	movpoints.row(i) = (T2 * movpoints.row(i).t()).t();
	srcpoints.row(i) = (T1 * srcpoints.row(i).t()).t();
		
}
	
movpoints.col(0) -= movX;	
movpoints.col(1) -= movY;	

srcpoints.col(0) -= srcX;	
srcpoints.col(1) -= srcY;	
	
Mat N,P1,P2;
multiply(movpoints.col(0), movpoints.col(1) ,N);
pow(movpoints.col(0),2,P1);	
pow(movpoints.col(1),2,P2);	
double beta =  0.5 * std::atan ( sum(N )[0]	/ (sum(P1)[0] - sum(P2)[0] ) ) ;

multiply(srcpoints.col(0), srcpoints.col(1) ,N);
pow(srcpoints.col(0),2,P1);	
pow(srcpoints.col(1),2,P2);	
double alpha = 0.5 * std::atan ( sum(N )[0]	/ (sum(P1)[0] - sum(P2)[0] ) ) ;	
	
double theta = alpha - beta;

R.at<float>(0,0) = cos(theta);	
R.at<float>(0,1) = -sin(theta);	
R.at<float>(1,0) = sin(theta);	
R.at<float>(1,1) = cos(theta);	

T2.at<float>(0,2) *= -1;
T2.at<float>(1,2) *= -1;
	
return T2 * R * T1;
	
}
void register_fram_pitch(Mat refframe, Mat movingframe,vector<float> & signal1,vector<float> & signal2,vector<float> & signal3, vector<float> &signal4, vector<float> &signal5,vector<float> &signal6)			// refframe = previous / static frame === movingframe = next / moving frame
{	
	
	Mat warp_matrix;
	Mat refgray,movgray;
	cvtColor(refframe, refgray,CV_BGR2GRAY);
	cvtColor(movingframe, movgray,CV_BGR2GRAY);
	 
	vector<Point2f> dstPf;
	vector<uchar> status;
	vector<float> err;
	vector<Point2f> srcPf;
		
	goodFeaturesToTrack(refgray, srcPf, 200, 0.01, 30);
	if(!srcPf.size()){return;} //movingframe; }
	
	calcOpticalFlowPyrLK(refgray,movgray,srcPf, dstPf,status,err);
	Mat mvgpts; Mat srcpts;
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){ Mat mvtp = Mat::zeros(1,2,CV_32F); mvtp.at<float>(0) = dstPf[i].x; mvtp.at<float>(1) = dstPf[i].y; mvgpts.push_back(mvtp); }};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){ Mat sctp = Mat::zeros(1,2,CV_32F); sctp.at<float>(0) = srcPf[i].x; sctp.at<float>(1) = srcPf[i].y; srcpts.push_back(sctp);	}};
		
	Mat warp_m;
	
//	cout<<"\n mvgpts size::"<<mvgpts.size()<<"\n";
	
	if( mvgpts.cols==0 || srcpts.cols == 0 )
	{
	return;
	}
	else
	{
		
	warp_m = get_pitch_warp(mvgpts,srcpts);
	warp_m.at<float>(0,2) = 0;

	/*	
	warp_m = get_affine_matrix(mvgpts,srcpts);
	warp_m.at<float>(0,0) = 1;
	warp_m.at<float>(0,1) = 0;
	warp_m.at<float>(1,0) = 0;
	warp_m.at<float>(1,1) = 1;
	warp_m.at<float>(0,2) = 0;
	cout<<"	"<<warp_m.at<float>(1,2)<<",";	*/
		
	}
	signal1.push_back(warp_m.at<float>(1,2));		
	signal2.push_back(warp_m.at<float>(0,0));		
	signal3.push_back(warp_m.at<float>(0,1));		
	signal4.push_back(warp_m.at<float>(1,0));		
	signal5.push_back(warp_m.at<float>(1,1));		
	signal6.push_back(warp_m.at<float>(0,2));		
	
	Mat warped_frame;// = warp_frame(movingframe, warp_m);
	warp_m.convertTo(warp_m,CV_32F);
//	warpAffine(movingframe,warped_frame,warp_m, movingframe.size());
//	warpPerspective(movingframe,warped_frame,warp_m, movingframe.size());
//	return warped_frame;

}

vector<float> getSimpleExpWindow(int winsize,float alpha = 0.5)
{
vector<float> window;
int filtS = winsize*2;	
alpha = 2/(float(filtS) + 1);
	

float s = 0;
for(int i = filtS; i>= 1;  i--)
{
window.push_back(alpha*pow((1-alpha),1-i));
s+= window[window.size() - 1];	
}

for(int i = filtS; i>= 1;  i--)
{
window[i] / s;
}
	
return window;	
}

vector<float> lpfilter(vector<float> signal)
{
	vector<float> kernel = getSimpleExpWindow(7);
	
	/*
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	kernel.push_back(0.1);
	*/
	
	vector<float> result;
	
	for(int i = 0; i< signal.size() + kernel.size() + 1; i++) {result.push_back(0);}
	
	for(int i = 0; i < signal.size(); i++)
	{
		for(int j = 0; j< kernel.size(); j++)
		{
		result[i+j] += signal[i] * kernel[j];						
		}
	}
	
	return result;
}

void stab_video_pitch_values(string inputPath)
{
string filnam;
string temname(inputPath);


	
size_t brck = inputPath.find_last_of("/");
size_t pnt = inputPath.find_last_of(".");

temname.replace(brck+1,pnt-1,"temp");
// filnam = inputPath.substr(brck+1,pnt - brck - 1);
					
	
int result  = rename(inputPath.data(),temname.data());
if(result != 0) {cout<<"\nRENAMING FILE FAILED\n"; exit(1); }

VideoCapture cap;
cap.open(temname.data());
	
VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open(inputPath.data(),codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));
	

		
Mat F;
 cap>>F;
vector<float> signal;	
vector<float> signal1;	
vector<float> signal2;	
vector<float> signal3;	
vector<float> signal4;
vector<float> signal5;
vector <int> hor;	
vector <int> horc;	
int i = 0;	
int horiY;
while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))
{
	
Mat K;
cap>>K;	
// register_fram_pitch(F,K,signal,signal1, signal2, signal3, signal4);
	
register_fram_pitch(F,K,signal,signal1, signal2, signal3, signal4,signal5);
K.copyTo(F);
i++;
}
	cap.set(CV_CAP_PROP_POS_FRAMES,0);
	int win_size = 7;
	vector<float> res = lpfilter(signal);	
	vector<float> res1 = lpfilter(signal1);	
	vector<float> res2 = lpfilter(signal2);	
	vector<float> res3 = lpfilter(signal3);	
	vector<float> res4 = lpfilter(signal4);	
	vector<float> res5 = lpfilter(signal5);	
	

	
	Mat t;


//for(int i = win_size - 1 ; i< signal.size() + (win_size) - 1; i++)
for(int i = 0 ; i< res.size() + (win_size) - 1; i++)
	{
			
		Mat cam_m = Mat::eye(3,3,CV_32F);	
		cam_m.at<float>(1,2) = res[i];
		
//		cam_m.at<float>(0,0) = res1[i];
//		cam_m.at<float>(0,1) = res2[i];
//		cam_m.at<float>(1,0) = res3[i];
//		cam_m.at<float>(1,1) = res4[i];
//		cam_m.at<float>(0,2) = res5[i];
		
		Mat CA;
		cap>>CA; if(CA.rows == 0 || CA.cols == 0 ){cap.release(); remove(temname.data());  return ;}
	warpPerspective(CA,CA,cam_m,CA.size());
		wir<<CA;
	}
cap.release();		
remove(temname.data());
}

vector<float> getDistancePara(vector<int> yvalues, float ymid = 1080 /2,float s = 0.15, float Rx = 1920,float rh = 3.3, float D = 500, float FOV = 60, float V = 1080)
{
 float diff,h,N; 
vector<float> areaT;	
	
	int J = 0; 
	
	for(int i = 0; i< yvalues.size(); i++)
	{

		if(yvalues[i] > V )
		{
		continue;
		}
		
	diff = yvalues[i] - ymid; 
	diff = (FOV / V) * diff;
	diff = (CV_PI / 180)* diff; 	
	diff = tan(diff);
	diff = 2/diff;	
	
	h = (D*D) / (D - rh);	
	h += D;
	N = (s*Rx)* (h - diff) / h; // diameter
	N = N / 2;					// radius
		
		areaT.push_back(N*N*CV_PI);

		
	}
	
	return areaT;
}

void running_average(string inputPath, string& morphedPath,int k = 5)
{
	
size_t brck = inputPath.find_last_of("/");
size_t pnt = inputPath.find_last_of(".");

morphedPath = inputPath;
morphedPath.replace(pnt,morphedPath.length(),"_temporal.avi");
	

VideoCapture cap;
cap.open(inputPath.data());

VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open(morphedPath.data(),codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));

int tot = cap.get(CV_CAP_PROP_FRAME_COUNT);
	
	
for(int i = 0; i< tot - k; i++)
{
	Mat frame = Mat::zeros(Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT)),CV_8UC3);
	cap.set(CV_CAP_PROP_POS_FRAMES,i);
						   
	for(int j = 0; j < k; j++)
	{
		Mat frat;
		cap>>frat;
		bitwise_or(frat,frame,frame);
	}
	wir<<frame;
}
	for(int i = tot - k; i<tot; i++)
	{
	Mat frame = Mat::zeros(Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT)),CV_8UC3);
	cap.set(CV_CAP_PROP_POS_FRAMES,i);
						   
	for(int j = 0; j < k; j++)
	{
		Mat frat;	
		cap>>frat;
		
		if(frat.rows == 0)
		{
                    cap.release();
                    return ;
			
		}
		bitwise_or(frat,frame,frame);
	}
	wir<<frame;
	
	}
	
	cap.release();
	// handle frame sizes after N - K

}

Mat MMSF(Mat img)
{

Mat G = Mat::zeros(3,3,CV_32F);
G.at<float>(0,0) = 1;	G.at<float>(0,0) = 1.1;	G.at<float>(0,0) = 1;
G.at<float>(0,0) = 1.1;	G.at<float>(0,0) = 1.6;	G.at<float>(0,0) = 1.1;
G.at<float>(0,0) = 1;	G.at<float>(0,0) = 1.1;	G.at<float>(0,0) = 1;

G /= 10;
	
Mat MA = Mat::ones(7,7,CV_32F); G.release();
MA /= 49;

Mat BS ; filter2D(img, BS,-1,MA); MA.release();
	
return (img - BS);	
}

void MMSF_process(string inputPath ,string& morphedPath)
{

size_t brck = inputPath.find_last_of("/");
size_t pnt = inputPath.find_last_of(".");

morphedPath = inputPath;
morphedPath.replace(pnt,morphedPath.length(),"_MSF.avi");
	
VideoCapture cap;
cap.open(inputPath.data());

VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open(morphedPath.data(),codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));

int tot = cap.get(CV_CAP_PROP_FRAME_COUNT);
	
for(int i = 0;i< tot; i++)
{
Mat mm;
cap>>mm;
mm = MMSF(mm);
wir<<mm;
}

	cap.release();
	// handle frame sizes after N - K


}


void horizon_remover(string inputPath)
{
		
cout<<"\n Removing Horizon from video file:: "<<inputPath.data()<<"\n";	
string filnam;
string temname(inputPath);

size_t brck = inputPath.find_last_of("/");
size_t pnt = inputPath.find_last_of(".");

temname.replace(brck+1,pnt-1,"temp");
// filnam = inputPath.substr(brck+1,pnt - brck - 1);
					
	
int result  = rename(inputPath.data(),temname.data());
if(result != 0) {cout<<"\nRENAMING FILE FAILED\n"; exit(1); }

VideoCapture cap;
cap.open(temname.data());
	
VideoWriter wir;
int codec = CV_FOURCC('M','J','P','G');
wir.open(inputPath.data(),codec ,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));
	
Mat fr;
cap>>fr;
fr = horizon_detect(fr);	 wir<<fr;
	
int i = 1;

	while (i < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
	
	Mat fra1;
	cap>>fra1;
	fra1 = horizon_detect(fra1);
	wir<<fra1;
	i++;
		
	}

	cout<<"\n InputPath Name:"<<inputPath.data()<<"\n";
	
    cap.release();
remove(temname.data());
}

void regst4(Mat src,Mat dst,Mat &U,const int warp_mode) // src:: reference/template frame dst::inst frame
{
// Convert images to gray scale;
Mat im1_gray, im2_gray;
	
cvtColor(src, im1_gray, CV_BGR2GRAY);
cvtColor(dst, im2_gray, CV_BGR2GRAY);
 
// Define the motion model
// const int warp_mode = MOTION_AFFINE;
 
// Set a 2x3 or 3x3 warp matrix depending on the motion model.
Mat warp_matrix;
 
// Initialize the matrix to identity
if ( warp_mode == MOTION_HOMOGRAPHY )
    warp_matrix = Mat::eye(3, 3, CV_32F);
else
    warp_matrix = Mat::eye(2, 3, CV_32F);
 
// Specify the number of iterations.
int number_of_iterations = 200;
 
// Specify the threshold of the increment
// in the correlation coefficient between two iterations
double termination_eps = 1e-5;
 
// Define termination criteria
TermCriteria criteria (TermCriteria::COUNT+TermCriteria::EPS, number_of_iterations, termination_eps);
 
// Run the ECC algorithm. The results are stored in warp_matrix.
findTransformECC(
                 im1_gray,
                 im2_gray,
                 warp_matrix,
                 warp_mode,
                 criteria
             );
 
// Storage for warped image.
 
if (warp_mode != MOTION_HOMOGRAPHY)
    // Use warpAffine for Translation, Euclidean and Affine
    warpAffine(dst,U, warp_matrix, src.size(), INTER_LINEAR + WARP_INVERSE_MAP);
else
    // Use warpPerspective for Homography
    warpPerspective (dst, U, warp_matrix, src.size(),INTER_LINEAR + WARP_INVERSE_MAP);
}

int otsu_tval( Mat img )
{
	Mat thres;   int t,threshold1,pixelval;   float sum=0.0,wb=0.0,wf=0.0,mb,mf,sumb=0.0,max=0.0,between;  int hist[256];
	 // Query for the frame
	Mat gray;


	cvtColor(img,gray,CV_BGR2GRAY); // Convert RGB image to Gray
	

	memset(hist, 0, 255);           //computes the histogram for the image
//	normalize(hist,hist,0,255,NORM_MINMAX);
	
	for (int i=0; i<img.rows; i++)
		for (int j=0;j<img.cols;j++) {
			pixelval = gray.at<uchar>(i,j);
			hist[pixelval] += 1;
		}

	for(int i = 0; i < 256; ++i)
		sum += i*hist[i];

	for (t = 0; t < 256; t++) {// Implementation of OTSU algorithm
		wb += hist[t];
		if(wb==0)
			continue;
		wf= img.cols*img.rows-wb;
		if(wf==0)
			break;
		sumb += (float)(t*hist[t]);
		mb = sumb/wb;
		mf = (sum-sumb)/wf;
		between = (float)wb*(float)wf*(mb-mf)*(mb-mf);

		if (between > max)	{
			threshold1 = t;
			max = between;
		}
	}

	
	return threshold1;

}

void CLAH( Mat bgr_image,Mat &image_clahe)
{
Mat lab_image;
	int flag = 0;
	if( bgr_image.channels() == 1)
	{
		flag = 1;
	cvtColor(bgr_image,lab_image,CV_GRAY2BGR);
	cvtColor(lab_image,lab_image,CV_BGR2Lab);
		
	}
	else
	{
	lab_image = bgr_image.clone();
    cvtColor(bgr_image, lab_image, CV_BGR2Lab);
	}
	
    // Extract the L channel
     vector< Mat> lab_planes(3);
     split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
     Ptr< CLAHE> clahe =  createCLAHE();
    clahe->setClipLimit(4);
     Mat dst;
    clahe->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
     merge(lab_planes, lab_image);
	

   // convert back to RGB
//	image_clahe = lab_image.clone();

    cvtColor(lab_image, image_clahe, CV_Lab2BGR);
	
	if (flag == 1)
	{
	cvtColor(image_clahe,image_clahe,CV_BGR2GRAY);
	}
}

//Usage B1::operator(image, output_Maskgroundmask) // Learning rate?? Confirm with previous matlab implementation.

Mat corr_hist(Mat M)
{
	vector<Mat> ch(3);
	vector<Mat> ch0(3);
	
	if (M.channels() == 1 )
	{
	
	Mat img;

	cvtColor(M,img,CV_GRAY2RGB);
	cvtColor(img,img,CV_RGB2HSV);
		
	split(img,ch0);
	equalizeHist(ch0[2],ch0[2]);
	CLAH(ch0[2],ch0[2]);

	img = ch0[2]; 
	vector<Mat>().swap(ch0);
		return img;
	}
	else
	{
		
	Mat img;
	img = M;
	cvtColor(img,img,CV_BGR2HSV);
	split(img,ch);
	equalizeHist(ch[2],ch[2]);
		
	CLAH(ch[2],ch[2]);
//	Mat KK = Mat(M.size(),CV_8UC3);
	Mat KK;
// cout<<"CH SIZE:::"<<ch.channels();

	merge(ch,KK);  vector<Mat>().swap(ch);
	cvtColor(KK,M,CV_HSV2BGR);
	return M;
	
	}
		

	
}

Mat corr_img(Mat img)
{
	// "channels" is a vector of 3 Mat arrays:
	vector<Mat> channels(3);
	
	// split img:
	split(img, channels);
	
	// get the channels (dont forget they follow BGR order in OpenCV)
	
	equalizeHist(channels[0],channels[0]);
	equalizeHist(channels[1],channels[1]);
	equalizeHist(channels[2],channels[2]);
	
	CLAH(channels[0],channels[0]);
	CLAH(channels[1],channels[1]);
	CLAH(channels[2],channels[2]);
		
	merge(channels,img);  vector<Mat>().swap(channels);
	return img; 
	
}


int weights3[3][3] = {{1, 8, 64},{ 2, 16, 128},{ 4, 32, 256}};

int offset(Mat BWin, int numRows, int numCols,int r, int c) 
{

    int minR, maxR, minC, maxC;
    int rr, cc;
	int result = 0;
    
    /* Figure out the neighborhood extent that does not go past */
    /* image boundaries */
    
	if (r == 0) {
        minR = 1;
    } else {
        minR = 0;
    }
    if (r == (numRows-1)) {
        maxR = 1;
    } else {
        maxR = 2;
    }
    if (c == 0) {
        minC = 1;
    } else {
        minC = 0;
    }
    if (c == (numCols-1)) {
        maxC = 1;
    } else {
        maxC = 2;
    }
	int t;
    for (rr = minR; rr <= maxR; rr++) {
        for (cc = minC; cc <= maxC; cc++) {
        
			t = BWin.at<uchar>(r+rr-1,c+cc-1) != 0 ;
			result += weights3[rr][cc] * t;
		
		}
    }

    return result;
}
void applylut(Mat BWin, Mat& BWout, Mat lut) // 8-bit applylut function scaled [ *255 at the end ]
{
    
    int numRows, numCols;

	BWout = Mat(BWin.size(),BWin.type(),Scalar(0));

    numRows = BWin.rows;
    numCols = BWin.cols;

    for (int c = 0; c < numCols; c++) {
        for (int r = 0; r < numRows; r++) {
           
			BWout.at<uchar>(r,c) = lut.at<unsigned int>( offset(BWin,numRows, numCols,r,c) );
		
		
		}
    }
		
//	imshow("RESULT",BWout);
//	waitKey();

	BWout = BWout*255;
}
void remove(Mat src,Mat &dst)
{

	int lut1[] = {0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,1,1,1,1,1,1,1,1 ,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ,1,1,1,1,1,1,1,1,1,1,1,1 ,1,1,1,1,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,1,1,1,1 ,1,1,1,1,1,1,1,1,1,1,1,1 ,0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,1,1,1,1,1,1,1,1 ,1,1,1,1,1,1,1,1,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0,0,0 ,1,1,1,1,1,1,1,1,1,1,1,1 ,1,1,1,1,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0 ,0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,1,1,1,1,1,1,1,1 ,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ,1,1,1,1,1,1,1,1,1,1,0,0 ,1,1,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,1,1,1,1 ,1,1,1,1,1,1,1,1,1,1,1,1 ,0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,1,1,1,1,1,1,1,1 ,1,1,1,1,1,1,1,1,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0,0,0 ,1,1,1,1,1,1,1,1,1,1,1,1 ,1,1,1,1,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,1,1,1,1 ,1,1,1,1,1,1,1,1,1,1,1,1 ,0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,1,1,1,1,1,1,1,1 ,1,1,1,1,1,1,1,1,0,0,0,0 ,0,0,0,0,0,0,0,0,0,0,0,0 ,1,1,1,1,1,1,1,1,1,1,0,0 ,1,1,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,0,0,0,0,1,1,1,1 ,1,1,1,1,1,1,1,1,1,1,1,1 ,0,0,0,0,0,0,0,0,0,0,0,0 ,0,0,0,0,1,1,1,1,1,1,1,1 ,1,1,0,0,1,1,0,0};
	// MATLAB REMOVE
	Mat lut = Mat(1,512,CV_8UC1,&lut1);
	applylut(src,dst,lut);
	dst*=255;
		
}
void bridge(Mat src,Mat&dst)
{
int lut1[]={0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
Mat lut = Mat(1,512,CV_8UC1,&lut1);
applylut(src,dst,lut);
dst *=255;
	
}
void diag(Mat src,Mat&dst) // prerequisite for thicken [not scaled] ---> output not multiplied by 255
{
	
int lutdiag[]={0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,0,0,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
Mat lut = Mat(1,512,CV_8UC1,&lutdiag);
applylut(src,dst,lut);
	
}
Mat thicken(Mat src)
{
	int lutis[] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	Mat lutiso = Mat(1,512,CV_8UC1,&lutis);
	int lutdil[] = {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	Mat lutdilate = Mat(1,512,CV_8UC1,&lutdil);
	int lutsing[] = { 0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	Mat lutsingle = Mat(1,512,CV_8UC1,&lutsing);
	Mat tsrc;
	applylut(src,tsrc,lutiso);  // tsrc  = iso

	if(countNonZero(tsrc) > 0)
	{
		Mat growmaybe,oneneighbor;
		applylut(tsrc,growmaybe,lutdilate); // DILATE VIA LOOKUP TABLE
		applylut(src,oneneighbor,lutsingle);
		bitwise_and(oneneighbor,growmaybe,growmaybe);
		bitwise_or(src,growmaybe,src);
	}
	
int R,C;  /// R === P:::C === Q
R = src.rows; C = src.cols;
Mat c = Mat::ones(src.rows+4,src.cols+4,src.type());
Mat not_a;
bitwise_not(src,not_a);
not_a.copyTo(c( Rect(2,2,C,R) ));
Mat cc;
thin(c,cc);
Mat d;
diag(cc,d);

Mat notcc,notc;
bitwise_not(cc,notcc);
bitwise_and(c,notcc,notcc);
bitwise_and(notcc,d,d);
	
bitwise_or(d,cc,c);

//    c(1:P,1:2) = 1;	P[R],Q[C]		%% Run for loops for these
c( Rect(0,0,2,R) ).setTo(Scalar(1));

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// c(1:P,(Q-1):Q) = 1;		%% Use equality conditions
c( Rect(C-2,0,2,R) ).setTo( Scalar(1) );

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// c(1:2,1:Q) = 1;
c( Rect(0,0,C,2) ).setTo( Scalar(1) );

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
//   c((P-1):P,1:Q) = 1;
c(Rect(0,R-2,C,2)).setTo(Scalar(1));
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	

// c = ~c(3:(P-2),3:(Q-2));
bitwise_not(c,notc);
Mat result = Mat(src.size(),notc.type());
notc( Rect(2,2,C,R) ).copyTo(result);
return result;

}
void closebw(Mat src,Mat &dst)
{
int lutero[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
int lutdil[] = {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
Mat luterode =  Mat(1,512,CV_8UC1,&lutero);
Mat lutdilate =  Mat(1,512,CV_8UC1,&lutdil);

applylut(src,dst,lutdilate);
applylut(dst,dst,luterode);
dst *=255;
	
}
void erodebw(Mat src,Mat & dst)
{
int lutero[]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
Mat luterode =  Mat(1,512,CV_8UC1,&lutero);
applylut(src,dst,luterode);
dst *=255;
}
void dilatebw(Mat src,Mat & dst)
{
int lutdil[] = {0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
Mat lutdilate =  Mat(1,512,CV_8UC1,&lutdil);
applylut(src,dst,lutdilate);
dst *= 255;
}
Mat bwareaopen(Mat binary,double thresh = 100) // remove elements lesser than area thresh
{
	normalize(binary,binary,0,1,NORM_MINMAX);
	vector <  vector< Point2i> > blobs;
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

             Rect rect;
             floodFill(label_image,  Point(x,y), label_count, &rect, 0, 0, 4);

             vector < Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back( Point2i(j,i));
                }
            }
			
			if( contourArea(blob,false) < thresh)
			{
			continue;
			}
			
            blobs.push_back(blob);

            label_count++;
        }
    }

//	 label_image.release();
     Mat output1 =  Mat::zeros(binary.size(), CV_8UC3);
     Mat output =  Mat::zeros(binary.size(), CV_8UC1);

    for(size_t i=0; i < blobs.size(); i++) {

		for(size_t j=0; j < blobs[i].size(); j++) {
            
			int x = blobs[i][j].x;
            int y = blobs[i][j].y;
			
            output1.at<Vec3b>(y,x)[0] = 255;
            output1.at<Vec3b>(y,x)[1] = 255;
            output1.at<Vec3b>(y,x)[2] = 255;
			
        }
  	
	}
	cvtColor(output1,output,CV_BGR2GRAY); output1.release();

	return output;
} 

Mat bwareaclose(Mat binary,double thresh = 100) // remove elements greater than area thresh
{
	normalize(binary,binary,0,1,NORM_MINMAX);
	vector <  vector< Point2i> > blobs;
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

             Rect rect;
             floodFill(label_image,  Point(x,y), label_count, &rect, 0, 0, 4);

             vector < Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back( Point2i(j,i));
                }
            }
			
			if( contourArea(blob,false) > thresh)
			{
			continue;
			}
			
            blobs.push_back(blob);

            label_count++;
        }
    }

//	 label_image.release();
     Mat output1 =  Mat::zeros(binary.size(), CV_8UC3);
     Mat output =  Mat::zeros(binary.size(), CV_8UC1);

    for(size_t i=0; i < blobs.size(); i++) {

		for(size_t j=0; j < blobs[i].size(); j++) {
            
			int x = blobs[i][j].x;
            int y = blobs[i][j].y;
			
            output1.at<Vec3b>(y,x)[0] = 255;
            output1.at<Vec3b>(y,x)[1] = 255;
            output1.at<Vec3b>(y,x)[2] = 255;
			
        }
  	
	}
	cvtColor(output1,output,CV_BGR2GRAY); output1.release();

	return output;
}

int h_global = 300;
int nmixtures_global = 3;
double BG_ratio_global = 0.75;

// B2 = new BackgroundSubtractorMOG(int h = 40, int nmixtures = 3, double backgroundRatio = 0.75);  
int globy = 0;


/*

       bH = obj.detector.step(eframe); 
        bH2 = obj.detector2.step(frame);
        
        bH11 = eframe.*repmat(uint8(bH),[1,1,3]);
              
        hsframe = rgb2hsv(bH11);
        hsframe2 = rgb2hsv(corr_hist(bH11));
    
        tem_fram11 = hsframe(:,:,3) > graythresh(hsframe); %% 0.16 previous 0.22 maximum
        tem_fram22 = hsframe2(:,:,3) > graythresh(hsframe2); %% 0.44 previous 0.55 maximum
   
          tem_fram1 = adaptivethreshold(hsframe(:,:,3),200,0);
          tem_fram2 = adaptivethreshold(hsframe2(:,:,3),200,0);
      
          tem_fram1 = tem_fram1 & tem_fram11;
          tem_fram2 = tem_fram2 & tem_fram22;
       
        tem_fram1 = bwmorph(tem_fram1,'remove'); %% 0.16 previous 0.22 maximum
        tem_fram2 = bwmorph(tem_fram2,'remove'); %% 0.44 previous 0.55 maximum
       
        tem_fram1 = bwmorph(tem_fram1,'close'); %% 0.16 previous 0.22 maximum
        tem_fram2 = bwmorph(tem_fram2,'close'); %% 0.44 previous 0.55 maximum
       
        tem_fram1 = bwmorph(tem_fram1,'thicken'); %% 0.16 previous 0.22 maximum
        tem_fram2 = bwmorph(tem_fram2,'thicken'); %% 0.44 previous 0.55 maximum
      
        tem_fram1 = bwmorph(tem_fram1,'bridge'); %% 0.16 previous 0.22 maximum
        tem_fram2 = bwmorph(tem_fram2,'bridge'); %% 0.44 previous 0.55 maximum
      
	  	tem_fram1 = bwareaopen(tem_fram1,100); %% 0.16 previous 0.22 maximum
        tem_fram2 = bwareaopen(tem_fram2,100); %% 0.44 previous 0.55 maximum
        
		tem_fram1 = bwmorph(tem_fram1,'close');
        tem_fram2 = bwmorph(tem_fram2,'close');
        

*/

// pre_bin saved in vid.cpp

Scalar Colors[]={Scalar(255,0,0),Scalar(0,255,0),Scalar(0,0,255),Scalar(255,255,0),Scalar(0,255,255),Scalar(255,0,255),Scalar(255,127,255),Scalar(127,0,255),Scalar(127,0,127)};


string get_filnam(const char X[])
{
string ch = X;
string h = ch.substr( 0,ch.length() - 4 );
//h.append(sizeof("_result"),"_result");
h+="_result";
// h.push_back("_result");
h.append( ch.substr(ch.length() - 4,4) );
return h;

}


int comp_fram(int curr_fram,vector<int> list)
{
int flag = 0;
cout<<"\nCurrent Frame ID::"<<curr_fram;
	for(int i = 0;i<list.size();i++)
	{
		cout<<"\n list[i]::"<<list[i];
		
		if(curr_fram == list[i])
		{
		cout<<"\n Returned Value::"<<list[i]<<" at ::"<<i<<"|||";	
		return i;
		}
		
	}
	
	return 99999;
	
}

Mat ret_rect(Mat src,int UI[])
{

	/*
	
	function II = ret_rect(I,x)
TIM = size(x,1);
[M,N,no] = size(I);
mask = zeros(M,N,no);
for p = 1:TIM
X = x(p,1);
Y = x(p,2);
xwidth = x(p,3);
yheight = x(p,4);
mask(Y:Y+yheight,X:X+xwidth) = 1;
end

II = I.*mask;

	*/

	Mat dst = Mat::zeros(src.size(),CV_8UC1);
	dst(Rect(UI[0]-1,UI[1]-1,UI[2],UI[3])).setTo(Scalar(1));
	bitwise_and(dst,src,dst);
	
	// multiply(dst,src,dst);
	return dst;
}

void ret_mask_frame(Mat &frame,Mat Mask)
{
	
vector<Mat> framel(3);
split(frame,framel);
	
bitwise_and(framel[0],Mask,framel[0]);
bitwise_and(framel[1],Mask,framel[1]);
bitwise_and(framel[2],Mask,framel[2]);

merge(framel,frame);
	
}

int training_frames = 200;


void	processing(Ptr<IFrameSource>	stabilizedFrames,	string	outputPath)
{
VideoWriter	writer;
Mat	stabilizedFrame;
int	nframes	=	0;
double	outputFps	=	25;
				//	for	each	stabilized	frame
while	(!(stabilizedFrame	=	stabilizedFrames->nextFrame()).empty())
				{
nframes++;
								//	init	writer	(once)	and	save	stabilized	frame
if	(!outputPath.empty())								{
if	(!writer.isOpened())																
writer.open(outputPath,VideoWriter::fourcc('X','V','I','D'),
outputFps,	stabilizedFrame.size());
writer	<<	stabilizedFrame;
								}
}

}

void stabilize_video(string inputPath)		// Put in full input path
{
cout<<"\n Stabilizing the footage from::"<<inputPath.data()<<"\n";
Ptr<IFrameSource>	stabilizedFrames;
				try
				{//	1-Prepare	the	input	video	and	check	it
string filnam;
string temname(inputPath);
					
size_t brck = inputPath.find_last_of("/");
size_t pnt = inputPath.find_last_of(".");

temname.replace(brck+1,pnt-1,"temp");
// filnam = inputPath.substr(brck+1,pnt - brck - 1);
					
int result  = rename(inputPath.data(),temname.data());
if(result != 0) {cout<<"\nRENAMING FILE FAILED\n"; exit(1); }

					
Ptr<VideoFileSource>	source	=	makePtr<VideoFileSource>(temname);
								cout	<<	"frame	count	(rough):	"	<<	source->count()	<<	endl;
//	2-Prepare	the	motion	estimator
//	first,	prepare	the	motion	the	estimation	builder,	RANSAC	L2
								double	min_inlier_ratio	=	0.1;
Ptr<MotionEstimatorRansacL2>	est	=	makePtr<MotionEstimatorRansacL2>(MM_AFFINE);
RansacParams	ransac	=	est->ransacParams();
								ransac.size	=	5; // (3)
								ransac.thresh	=	10;	// (5)
								ransac.eps	=	0.5;
								est->setRansacParams(ransac);
								est->setMinInlierRatio(min_inlier_ratio);
				//	second,	create	a	feature	detector
int	nkps	=	1000;
Ptr<GFTTDetector> feature_detector = GFTTDetector::create(nkps);
					
// Ptr<GoodFeaturesToTrackDetector>	feature_detector	=	makePtr<GoodFeaturesToTrackDetector>(nkps);
//	third,	create	the	motion	estimator
Ptr<KeypointBasedMotionEstimator>	motionEstBuilder	=	makePtr<KeypointBasedMotionEstimator>(est);
								motionEstBuilder->setDetector(feature_detector);
Ptr<IOutlierRejector>	outlierRejector	=	makePtr<NullOutlierRejector>();
								motionEstBuilder->setOutlierRejector(outlierRejector);
//	3)			prepare	the	stabilizer
StabilizerBase	*stabilizer	=	0;
//	first,	prepare	the	one	or	two	pass	stabilizer
								bool	isTwoPass	=	1;
								int	radius_pass	=	15;
								if	(isTwoPass)
								{
												//	with	a	two	pass	stabilizer
												bool	est_trim	=	false;
TwoPassStabilizer	*twoPassStabilizer	=	new	TwoPassStabilizer();
												twoPassStabilizer->setEstimateTrimRatio(est_trim);
												twoPassStabilizer->setMotionStabilizer(makePtr<GaussianMotionFilter>(radius_pass));												stabilizer	=	twoPassStabilizer;
								}
								else
								{
												//	with	an	one	pass	stabilizer
OnePassStabilizer	*onePassStabilizer	=	new	OnePassStabilizer();
												onePassStabilizer->setMotionFilter(makePtr<GaussianMotionFilter>(radius_pass));
												stabilizer	=	onePassStabilizer;
								}
								//	second,	set	up	the	parameters
								int	radius	=	15;
								double	trim_ratio	=	0;
								bool	incl_constr	=	false;
stabilizer->setFrameSource(source);
stabilizer->setMotionEstimator(motionEstBuilder);
								stabilizer->setRadius(radius);
								stabilizer->setTrimRatio(trim_ratio);
								stabilizer->setCorrectionForInclusion(incl_constr);
								stabilizer->setBorderMode(BORDER_REPLICATE);
								//	cast	stabilizer	to	simple	frame	source	interface	to	read	stabilized	frames
					stabilizedFrames.reset(dynamic_cast<IFrameSource*>(stabilizer));
//	4-Processing	the	stabilized	frames.	The	results	are	showed	and	saved.
//					outputPath = inputPath.append("stab_new.mp4");
// int res = rename(temname.data(),inputPath.data());
// if(res != 0){cout<<"\nRENAMING FAILED\n"; exit(1)}
					
processing(stabilizedFrames,inputPath);
remove(temname.data());
				}
				catch	(const	exception	&e)
				{
								cout	<<	"error:	"	<<	e.what()	<<	endl;
								stabilizedFrames.release();
								return;
				}
				stabilizedFrames.release();

}

void regst_optical_flow(Mat dst,Mat src, Mat& U) // src ~~ template ||||| dst ~~ Target, the image that you want it to become like		// LUCAS KANADE REGISTERATION
{
    
	Mat SRC,DST;
	
	if (src.channels() == 3 || src.channels() > 1){
	cvtColor(src,SRC,CV_BGR2GRAY);
	}
	else{
	SRC = src;
	}
	
	if (dst.channels() == 3 || dst.channels() > 1){
	cvtColor(dst,DST,CV_BGR2GRAY);
	}
	else{
	DST = dst;
	}
		

	vector<KeyPoint> srcP;
	vector<KeyPoint> srcP2;
	vector<KeyPoint> dstP;
	vector<Point2f> srcPf;
	vector<Point2f> srcPf2;
	vector<Point2f> dstPf;
	vector<Point2f> dstPf2;
	vector<uchar> status;
	vector<float> err;
	
	Mat sD,dD;
	

	
	goodFeaturesToTrack(SRC, srcPf, 200, 0.01, 30);

	
	calcOpticalFlowPyrLK(SRC,DST,srcPf, dstPf,status,err);
	
	for(int i = 0;i<srcPf.size();i++){	if(status[i]){srcPf2.push_back(srcPf[i]);	}	};
	for(int i = 0;i<dstPf.size();i++){	if(status[i]){dstPf2.push_back(dstPf[i]);	}	};
	


	Mat app; Mat H;
	
	if(srcPf2.size() == 0 || dstPf2.size() == 0)
	{	H = Mat::eye(3,3,CV_32F);	
	warpPerspective(dst,U,H,dst.size(),INTER_LINEAR);
	 return;
	}
	H = findHomography( srcPf2,dstPf2, CV_RANSAC ); /// Next Frame, Reference (Previous) Frame
	if(H.empty()){H = Mat::eye(3,3,CV_32F);}
	
warpPerspective(dst,U,H,dst.size(),INTER_LINEAR);
// warpAffine(dst, U,cont, dst.size());
}

void regis_video(string sname)
{
	
string temname(sname);
size_t brck = sname.find_last_of("/");
size_t pnt = sname.find_last_of(".");
temname.replace(brck+1,pnt-1,"temp");	
int result  = rename(sname.data(),temname.data());
	
VideoCapture cap;
cap.open(temname.data());
VideoWriter wir;
int codec = CV_FOURCC('X','V','I','D');
wir.open(sname.data(),codec,cap.get(CV_CAP_PROP_FPS),Size( cap.get(CV_CAP_PROP_FRAME_WIDTH) ,cap.get(CV_CAP_PROP_FRAME_HEIGHT) ));		
if(result != 0) {cout<<"\nRENAMING FILE FAILED\n"; exit(1); }
	
Mat dfram;
cap>>dfram;
resize(dfram,dfram,Size(640,360));

	while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
		
	Mat frame;
	Mat newfram;
	cap>>frame;
	resize(frame,frame,Size(640,360));
	regst_optical_flow(frame,dfram,newfram);
	frame.copyTo(dfram);
	wir<<newfram;
	cout<<".";
	}
cap.release();	
remove(temname.data());	
}

int trackID = 0;

struct detect
{
	Point2f pt;
	Rect bb;
	int age = 0;
	int invisible_age = 0;
	int invisible_flag = 0;
	int segmentID;
	int total_age = 0;
	vector<int> frameID;
	vector<int> visi_chain;
	vector<Point2f> trace;
	vector<int> pxcount;
	shared_ptr<TKalmanFilter> KalmanF;
	int tID = 0;
	vector<float> minboundradius;
//	vector<float> vari;
//	vector<float> secmoment;
	int rechecked_age = 0;
        int intime = 0;
	vector <int> width;
	vector <int> height;

	detect()
	{tID = trackID++;}
	
};

struct thold
{
	int minavg = 0,maxavg = 0;
	int minmax = 0, maxmax = 0;
};

thold getTholds(int seg, Point2f pt )
{
thold TH;
	
	int reg = 0;
	
	if(pt.y < 180)
	{
	reg = 0;
	}
	else
	{
	reg = 1;
	}
	
	if(seg%2 == 0 && reg == 0)				// TOP
	{
		TH.minavg = 9;
		TH.maxavg = 30;
		TH.minmax = 0;
		TH.maxmax = 300;
	}
	else if(seg%2 == 0 && reg == 1)						// BOTTOM
	{
	TH.minavg = 30;
	TH.maxavg = 60;
	TH.minmax = 0;
	TH.maxmax = 300;
	}
	else if(seg%2 == 1 && reg == 0)
	{
	TH.minavg = 60;
	TH.minavg = 100;
	TH.minmax = 300;
	TH.maxmax = 1000;
	}
	else
	{
		TH.minavg = 100;
		TH.maxavg = 500;	
		TH.minmax = 300;
		TH.maxmax = 1000;
	}
	
	return TH;
}

vector<detect> getParasNew(Mat &Mask,Mat& frame, int FRAMEID, int segmentID, float dt = 0.2, float accel_noise_mag = 0.5)   // 13 //
{
vector<detect> det;
Mat labimg;

int nolab = connectedComponents(Mask,labimg,8);
	
	for(int i = 0; i< nolab; i++)			// nunber of labels equal the number of objects found in the image
	{
		vector<Point> pnt;
		int compcount = 0;
		
		for(int I = 0; I< labimg.rows; I++)
		{
			for(int J = 0; J< labimg.cols; J++)
			{
		if(labimg.at<int>(I,J) == i)
		{
		pnt.push_back(Point(J,I));
		compcount++;
		}
			}
		}

	float rad;
		Point2f tt;
		minEnclosingCircle(Mat(pnt),tt,rad);

		
	
	
	Rect BR = boundingRect(Mat(pnt));
	int centx,centy;
	centx = (BR.x) + (BR.width)/2;
	centy = (BR.y) + (BR.height)/2;
	Point centi(centx,centy);
	/*
		float varin = 0;
		float momen = 0;
		float summ = 0;
		for(int k = 0; k < pnt.size(); k++)
		{
		Point PPP = centi - pnt[k];
		varin += sqrt( pow(PPP.x,2) + pow(PPP.y,2) );
			
			float xx = pow( centi.x - pnt[k].x,2);
			float yy = pow( centi.y - pnt[k].y,2);
			
			momen += xx*yy*(Mask.at<uchar>(pnt[k].y,pnt[k].x));
			summ += (Mask.at<uchar>(pnt[k].y,pnt[k].x));			
		}
		
		momen /= summ;
	*/			// SECOND MOMENT AND VARIANCE
		
		
	detect ddt;		
		
//	ddt.secmoment.push_back(momen);
	ddt.segmentID = segmentID;
	ddt.visi_chain.push_back(1);
//	ddt.vari.push_back(varin);	
	ddt.minboundradius.push_back(float(rad));
	ddt.pt.x = centx;
	ddt.pt.y = centy;
	ddt.trace.push_back(ddt.pt);
	ddt.bb = BR;
	ddt.age = 1;
	ddt.total_age = 1;
	ddt.invisible_age = 0;
	ddt.frameID.push_back(FRAMEID);
	ddt.KalmanF = make_shared<TKalmanFilter>(Point2f(centx,centy),dt,accel_noise_mag);
        ddt.pxcount.push_back(compcount);
	ddt.intime = 0;	
	ddt.width.push_back(BR.width);	
	ddt.height.push_back(BR.height);	
//	ddt.PTS.push_back(pnt2);	
	det.push_back(ddt);
	}
	
return det;	
}

vector<detect> paraUpdateNN(vector<detect> &det, vector<detect>& D,int &FrameID,int dist_thresh = 20,int mini_d_th = 7, double hist_thresh_low = 0.4, double hist_thresh_high = 0.6, int intime_thresh = 3, float age_frac_th = 0.3, int total_age_thresh = 70, int inv_th = 20, float color_thresh = 95)       // 14 //
{
		vector< vector<double> > CST(det.size(),vector<double>(D.size()) );
		vector<int> assign;
		vector<int> not_assigned_tracks;
		vector<int> not_assigned_detections;
		vector<int>::iterator it;
		
		
		for(int U = 0; U< det.size(); U++)
		{
			for(int V = 0; V< D.size(); V++)
			{
			Point d = det[U].pt - D[V].pt;	
			CST[U][V] = std::sqrt(d.x*d.x + d.y*d.y);			// 15 a.) //	//  the cost matrix	
			}
			
			if(det[i].invisible_age > 10)
                        {
                            det[i].invisible_flag = 1;                            
                        }
			
		}
		
		AssignmentProblemSolver APS;
		APS.Solve(CST,assign,AssignmentProblemSolver::optimal);                 // 15 b.) //
	
		for(int u = 0; u< assign.size(); u++)		// detect unassigned tracks
		{			
			if(CST[u][assign[u]] > dist_thresh || det[i].invisible_flag == 1 )
			{
			assign[u] = -1;
			}		
			
			// intime threshold condition removed (If goes invisible more than a few times than is unnecessary)			
		}
	
	
		for(int u = 0; u< assign.size(); u++)		// detect unassigned tracks
		{
			/// still missing the cost matrix threshold on distance add later
			
			if(assign[u] == -1)
			{
			not_assigned_tracks.push_back(u);
			}
			
		}
		
		for(int y = 0; y < D.size();y++ )			// detect unassigned detections
		{
		it = find(assign.begin(),assign.end(),y);
			if(it == assign.end())
			{
			not_assigned_detections.push_back(y);	
			}
		}
		
		for(int t = 0; t< assign.size(); t++)
		{
			det[t].KalmanF->GetPrediction();
			
			if(assign[t] != -1)                  // 15 c.) //
			{
						
				det[t].pt = det[t].KalmanF->Update(D[assign[t]].pt,1);
				det[t].bb = D[assign[t]].bb;
				det[t].pxcount.push_back( D[assign[t]].pxcount[ D[assign[t]].pxcount.size() - 1 ] );
				det[t].minboundradius.push_back( D[assign[t]].minboundradius[ D[assign[t]].minboundradius.size() - 1 ] );
				det[t].age++;
				det[t].total_age++;
				det[t].invisible_age = 0;
				det[t].width.push_back(  D[assign[t]].width[ D[assign[t]].width.size() - 1  ] );
				det[t].height.push_back(  D[assign[t]].height[ D[assign[t]].height.size() - 1  ] );
				det[t].visi_chain.push_back(1);													// det has been found again and visible in the next frame
				
			}
			else                                 // 15 e.) //
			{
			// could warrant for the creation of a invisible frameID in the future here	
			det[t].pt = det[t].KalmanF->Update(Point2f(0,0),0);
			if(det[t].invisible_age == 0){det[t].intime++;}
			det[t].visi_chain.push_back(0);					
			det[t].invisible_age++;
			det[t].total_age++;
			}
			
			det[t].frameID.push_back( FrameID );
			det[t].trace.push_back( det[t].pt );
			
		}
	
		vector<detect> lo;
	
		for(int i = 0; i< det.size(); i++)
		{	
			Point2f P;
			P = det[i].trace[0] - det[i].pt; // if(det[t].tID == 39941){ cout<<"\n Distance Criteria::"<< std::sqrt( (P.x * P.x) + (P.y * P.y) )<<"\n"; }
		 
			if(det[i].age > total_age_thresh)	// 15 f.) // 	/// Deletion
			{
			
			}
			else
			{
			lo.push_back(det[i]);
			}
		}
		det = lo;
	
		for(int t = 0; t< not_assigned_detections.size(); t++)		// 15 d.) //	// adding new detections
		{
			det.push_back(D[not_assigned_detections[t]]);
		}
		

	
	
	return det;
}

Mat getMaskInfo(vector<detect>& D,Mat frame)
{

	for(int i = 0; i< D.size(); i++)
	{
	stringstream conv;
	conv<< D[i].pxcount[ D[i].pxcount.size() - 1 ];
			int maxpxcount = 0, avgpxcount = 0;
			
				for(int n = 0; n< D[i].pxcount.size(); n++)
				{
				avgpxcount += D[i].pxcount[n];
					if(D[i].pxcount[n] > maxpxcount )
					{
					maxpxcount = D[i].pxcount[n];
					}
				}
			avgpxcount /= D[i].pxcount.size();
			stringstream con2;
			con2<< D[i].tID;
	putText(frame,con2.str(),D[i].pt,FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),1); 		/*Displays Track ID*/

	}
	
	return frame;
	
}

float thold_det(float centy,float Hr, float Vr, float rh,float vdeg,float s, float D)
{

	float X = vdeg / Vr;	
	X *= abs(centy - (Vr/2) );	
	X *= (CV_PI/180);	
	X = 2/(tan(X));
	
	cout<<"\n distance::"<<X<<"\n";
	
	float A = s*(Hr)*(2*D  - rh)*D;
	A /= ((2*D - rh)*D - X*(D - rh));

	cout<<"\n Area for threshold::"<<CV_PI * A * A /4 <<"\n";
	exit(0);
		return A;
}

float getStdD( vector<int> pixels,float average)
{
float stddev = 0;
	
	for(int i =0; i< pixels.size(); i++)
	{
			stddev += pow(pixels[i] - average,2);
		
	}

	stddev /= pixels.size();
	if(stddev){ stddev = sqrt(stddev); }else{stddev = 0;}
	
	return stddev;
	
}

float getStdDDEV(vector<Point2f> pnts)
{
float x = 0,y = 0;
	for(int i = 0; i < pnts.size(); i++)
	{
	
		pnts[i].x += x;
		pnts[i].y += y;
		
	}
	
	x /= pnts.size();
	y /= pnts.size();
	
	float s1 = 0,s2 = 0;
	for(int i = 0; i < pnts.size(); i++)
	{
	
		pnts[i].x -= x;
		s1 += pnts[i].x * pnts[i].x;
		pnts[i].y -= y;
		s2 += pnts[i].y * pnts[i].y;
	}

	s1 /= pnts.size();
	s2 /= pnts.size();
	
	s1 = sqrt(s1);
	s2 = sqrt(s2);
	
	s1 = sqrt( pow(s1,2) + pow(s2,2) );
	
	return s1;
}

vector<detect> renew_pross(string MSFPath, string inputPath,vector<detect> true_det,int segID)      // 19 //
{
	float dist_thresh = 10;
	float colo_thresh = 10;
	
	VideoCapture cap,capORI;
	cap.open(MSFPath.data());
	capORI.open(inputPath.data());
	
	Ptr<BackgroundSubtractorMOG2> B1 = createBackgroundSubtractorMOG2();
	B1->setNMixtures(7);
	
	int frameIID = 0;
	
	while ( cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT) )
	{
	
		Mat frame,bgr;
                vector<Vec3b> cvalues;
		cap>>frame;
		B1->apply(frame,bgr);                                                             // 19 a.) //
		vector<detect> det = getParasNew(bgr,frame, frameIID,segID);
		Mat frameORI;
		capORI>>frameORI;                                                                
		
                for(int i = 0; i< det.size(); i++)      // Pre-allocating color values per iteration to not do it multiple times every time a detection is found in frameID
                {
                    cvalues.push_back(frameORI.at<Vec3b>(det[i].pt.y, det[i].pt.x));                     // 19 b.) // 
                }
                
		for(int i = 0; i<true_det.size();i++ )
		{
		int fl = 0;
		int JVAL;
			for(int j = 0; j< true_det[i].frameID.size(); j++)                               // 19 c.) //
			{
				if(true_det[i].visi_chain[j])			// if true det is not invisible at this frameID 
					{
						if(frameIID == true_det[i].frameID[j])
						{
							fl = 1;
							JVAL = j;
						break;
						}
					}
			}
			
			if(fl)
			{
			
				Point2f fdp = true_det[i].trace[JVAL];
				Vec3b cval = frameORI.at<Vec3b>(fdp.y,fdp.x);

				
				for(int k = 0; k < det.size(); k++)
				{
					Point2f d = det[k].pt - fdp;
//					Vec3b CVAL = frameORI.at<Vec3b>(det[k].pt.y, det[k].pt.x);           // <---- This line quite intensive in itself not called over and over again
					Vec3b ni = cval - cvalues[k];
					
					if( ( sqrt(d.x*d.x + d.y*d.y) < dist_thresh ) && ( sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2]) < colo_thresh )  )        // 19 d.) //    
					{
					true_det[i].rechecked_age++;
					}					
				}
			}	

		} // at this stage have rechecked age for all true dets

		frameIID++;
	}
	
		
		vector<detect> ret_det;
		
		for(int t = 0; t< true_det.size(); t++)
		{
				if(true_det[t].rechecked_age > (true_det[t].age / 2))
				{
				ret_det.push_back(true_det[t]);
				}

		}
	
	cap.release();
        capORI.release();
	return ret_det;
}

vector <Mat> split_frame(Mat& frame)		// pass by reference to save space
{
	vector<Mat> FR;
	
	FR.push_back( frame( Rect(0,359,640,360) ) );					//0
	FR.push_back( frame( Rect(0,719,640,360) ) );					//1
	FR.push_back( frame( Rect(639,359,640,360) ) );					//2
	FR.push_back( frame( Rect(639,719,640,360) ) );					//3
	FR.push_back( frame( Rect(1279,359,640,360) ) );				//4
	FR.push_back( frame( Rect(1279,719,640,360) ) );				//5
		
	return FR;
}

vector<string> getSegmentList(string inputPath)
{
vector<string> allstreams;

string tem(inputPath);

size_t tm = tem.find_last_of(".");
	
	for(int i = 0; i<6; i++ )
	{
	
		stringstream ah;
		ah << i;
		
		string path("segment_");
		path.append( ah.str() );
		path.append( ".avi" );
		string tem2(inputPath);
		tem2.replace(tm,tem2.length(),path.data());		
		allstreams.push_back(tem2);	
	}
	
return allstreams;
}

void create_video_files(vector<string> &alist, string inputPath)
{
	VideoCapture cap;
	cap.open(inputPath.data());

	alist = getSegmentList(inputPath);
	
	vector<VideoWriter>	vwir;
	
	for(int i = 0; i< alist.size(); i++)
	{
	VideoWriter wir;
	int codec = CV_FOURCC('M','J','P','G');
	wir.open(alist[i].data(),codec ,cap.get(CV_CAP_PROP_FPS),Size( 640,360 ));
	vwir.push_back(wir);	
	}
	
	while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))
	{
	
		Mat ffr;
		cap>>ffr;
		vector<Mat> vf = split_frame(ffr);
		
			for(int i = 0; i< vf.size(); i++)
			{
			vwir[i] << vf[i];				
			}
		
	
	}
	
	cap.release();
}

void delete_video_files(vector<string> alist)
{

	for(int i = 0; i< alist.size(); i++)
	{
	remove(alist[i].data());
	}
	
}

void clear_memory()
{

	system(" sudo rm -r /tmp/* ");

}

void filepross(const char sname[],int f_ID,char nname[],int segmentID)
{
	::trackID = 0;
	string input = string(sname);
	string morphed,MSFPath;
	
ofstream of;
of.open("/media/ml/New Volume1/W_motion/new_tracker_BFTB.txt",ios::out | ios::ate | ios::app);      // 6 //
	

	running_average(string(sname),morphed);                                                     // 7 //
	MMSF_process(string(sname),MSFPath);                                                        
	
	
cout<<"\n sname::"<<sname<<" Frames Aligned by registration \n";
	int yflag = 0;
	int FRAMEID = 0; 
	
	Ptr<BackgroundSubtractorMOG2> B3 = createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractorMOG2> B4 = createBackgroundSubtractorMOG2();
	Ptr<BackgroundSubtractorMOG2> B5 = createBackgroundSubtractorMOG2();
	
	B3->setNMixtures(7);
	B4->setNMixtures(7);
	B5->setNMixtures(7);
	
	//some boolean variables for added functionality
	int morph_size = 2;
    
/*	Mat element = getStructuringElement( MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size + 1 ) );
	Mat element2 = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size + 1 ) );
	Mat element3 = getStructuringElement( MORPH_RECT, Size( 2*morph_size - 1, 2*morph_size - 1 ) );		*/
	Mat element4 = getStructuringElement( MORPH_ELLIPSE, Size( morph_size, morph_size ) );
//	Mat element5 = getStructuringElement( MORPH_RECT, Size( morph_size, morph_size ) );
	
	Mat diamond = Mat(5,5,CV_8U,Scalar(1));
	diamond.at<uchar>(0,0)= 0;
    diamond.at<uchar>(0,1)= 0;
	diamond.at<uchar>(1,0)= 0;
	diamond.at<uchar>(4,4)= 0;
	diamond.at<uchar>(3,4)= 0;
	diamond.at<uchar>(4,3)= 0;
	diamond.at<uchar>(4,0)= 0;
	diamond.at<uchar>(4,1)= 0;
	diamond.at<uchar>(3,0)= 0;
	diamond.at<uchar>(0,4)= 0;
	diamond.at<uchar>(0,3)= 0;
	diamond.at<uchar>(1,4)= 0;
	
	Mat ker = Mat::zeros(3,3,CV_32F);
	
	ker.at<float>(0,0) = 0;		ker.at<float>(0,1) = -0.25;	ker.at<float>(0,2) = 0;
	ker.at<float>(1,0) = -0.25;	ker.at<float>(1,1) = 2;		ker.at<float>(1,2) = -0.25;
	ker.at<float>(2,0) = 0;		ker.at<float>(2,1) = -0.25;	ker.at<float>(2,2) = 0;

	
	Mat thresholdImage;
	int N = 640;
	int M = 360;
	int count = 1;
	Mat eframe,frame,G1,G2;
	Mat Mask;
	Mat temeframe1,temeframe2;
	Mat efr1bg,efr2bg,efr3bg;

	int  Training_Sequence = 25;
	
	VideoCapture cap;
	

cout<<"\n Processed Filename:::"<<morphed.data()<<"\n";
	cap.open(morphed.data()); // sname is the name of the file in character format
	

		if(!cap.isOpened()){
			cout<<"ERROR ACQUIRING VIDEO FEED\n";
			getchar();
			return;
		}
	
	vector<Point2f> centers;
	vector<detect> true_det;
	
/*Previous Value*/	
		int lesser_age = 9;
		int avgpxthresh = 150;				// seems to be a crucial factor (Average dolphin detection for interpolated 320 * 180 to 640 *360 needs 200 pixels )
		int large_age = 30;			// Too low a value would cause non-detection of objects which last longer
/*50*/	int minpxthresh = 50;
		int max_movement = 8;
										// Would be a good fit for test videos but not for project ganga	
	int fra_win_size = 100;
	
//	while(FRAMEID < 100)
	int win_flag = 0;
	
	vector<detect> det;
	Mat prevf;
	cap>>prevf;
	resize(prevf,prevf,Size(640,360));
	
	while(cap.get(CV_CAP_PROP_POS_FRAMES) < cap.get(CV_CAP_PROP_FRAME_COUNT))	             // 9 //
	{
		
		
		cap>>frame;
			if(frame.cols == 0){cout<<"\n Reached end or error reading file!"; exit(1); }
			
		resize(frame,frame,Size(640,360));

			dilate( frame,temeframe1, diamond);
			morphologyEx( frame,temeframe2, MORPH_CLOSE, diamond);
			
			eframe = temeframe1 - temeframe2;	// morphologyEx( fram_list[i],eframe, MORPH_GRADIENT, element);// OpenCV version  // detimg =  imdilate(img,strel(str,num)) - imclose(img,strel(str,num)); // MatLab Version
			
			temeframe1.release();
			temeframe2.release();
		
			if( (eframe.cols < 1) && (eframe.rows< 1) )
			{
			
				cout<<"INSUFFICIENT ROWS AND COLS";
				return;
				
			}
		

			
			Mat efr1,efr2,efr3;
	
			Mat BG3,BG4,BG5,FMask; 			/// REMOVE THIS
		
//			normalize(eframe,efr1,0,255,CV_MINMAX);   // REDUCE COMPUTATIONS LATER BY REMOVING UNNECESSARY TERMs		
			homomor(eframe,efr2,ker);		
            

		B3->apply(eframe,BG3);                // 11 //
		B4->apply(efr2,BG4); 
 		bitwise_and(BG3,BG4,Mask);
		medianBlur(Mask,Mask,5);
		morphologyEx(Mask,Mask,MORPH_OPEN,element4);
		morphologyEx(Mask,Mask,MORPH_CLOSE,element4);
			
			
		int dist_thresh = 7;				// Bounding boxes must be within dist_thresh pixels of one another
		int area_thresh = 25;				// The area of the objects contained within the bounding boxes must be within area_thresh nbd of one another

		if(FRAMEID == 0)
		{
		det = getParasNew(Mask,frame,FRAMEID,segmentID);		    // 12 //			// First while loop iteration + first for loop one too 
		FRAMEID++;
		continue;
		}	
		vector<detect> D = getParasNew(Mask,frame, FRAMEID,segmentID);
		det = paraUpdateNN(det,D,FRAMEID);
		

		string imgpath("/media/ml/New Volume1/W_motion/Frame_BF_Data/");
		stringstream convert;
		stringstream seg; seg << segmentID;
		convert << FRAMEID;
		imgpath.append("Tracking_44971_");
		imgpath.append(convert.str());
		imgpath.append("_segmentID_");
		imgpath.append(seg.str());
		imgpath.append(".jpeg");
		int kswitch = imwrite(imgpath.data(),getMaskInfo(det,frame));
	

			thresholdImage.release();
			Mask.release();
		
		vector<detect> Dnew;
		/// Parameters defined here
		int invisi_age_thresh = 10;

		for(int i = 0; i< det.size(); i++)                                // 16 //
		{
		
			if(det[i].invisible_age)
			{													// 													(SURELY INVISIBLE OBJECTS)
				if(det[i].invisible_age > invisi_age_thresh)			// parameterize later
				{
				det[i].invisible_flag = 1;			// gone invisible wrt. window of frames
					//
/*							
		float avg_pxvelocity = 0;
				
					for(int y = 0; y< det[i].trace.size() - 1; y++)
				{
					Point2f po1;
					po1 = det[i].trace[y] - det[i].trace[y+1];
					float pxv = sqrt(  po1.x*po1.x + po1.y*po1.y   );
					avg_pxvelocity += pxv;	
					
				}
					avg_pxvelocity /= ( det[i].trace.size() - 1);	
*/				// Average Pixel Velocity ::::::: NOT NECESSARY NOW CAN BE USED LATER ON WHEN DISTANCE INFORMATION ASSIMILATED		
				
		int maxpxcount = 0;
		int avgpxcount = 0;
			
				for(int n = 0; n< det[i].pxcount.size(); n++)
				{
				avgpxcount += det[i].pxcount[n];
					if(det[i].pxcount[n] > maxpxcount )
					{
					maxpxcount = det[i].pxcount[n];
					}
				}
				avgpxcount /= det[i].pxcount.size();
				
				
				float rad = 0;
				for(int l = 0; l < det[i].minboundradius.size(); l++ )
				{
				
					rad += det[i].minboundradius[l];
					
				}
				rad /= det[i].minboundradius.size();
								
				int wid = 0;
				int hei = 0;
				for(int l = 0;l< det[i].width.size(); l++ )
				{
					wid += det[i].width[l];
					hei += det[i].height[l];
				}
				float aspect_ratio = float(hei /= det[i].height.size()) / float(wid /= det[i].width.size() ) ;
				
				
		double circly,ffactor;
		circly = ( (4*avgpxcount)/(CV_PI * rad*2*rad*2) ); 
			Point2f P;
				P = det[i].trace[0] - det[i].pt;
				float age_ratio = float(det[i].age) / float(det[i].total_age);


		if(segmentID%2 != 0)
		{
			if( (avgpxcount > 350) && (!(det[i].age < lesser_age)) &&  !( maxpxcount > avgpxthresh ) && (!(det[i].age > large_age)) && (  std::sqrt( (P.x * P.x) + (P.y * P.y) ) < max_movement  ) && (age_ratio  > 0.45) && (aspect_ratio == 1 || sqrt(pow(aspect_ratio - 1,2)) <= 0.5 ) && ( (circly*aspect_ratio > 0.5 )  && (circly*aspect_ratio < 1)  )  )          // 17 //
                            
                            // && (aspect_ratio == 1 || sqrt(pow(aspect_ratio - 1,2)) <= 0.5 ) (  std::sqrt( (P.x * P.x) + (P.y * P.y) ) < max_movement  ) && (age_ratio  > 0.45)  )// && (det[i].age / det[i].total_age  > 0.45) ) //&& (det[i].intime)) && (stddev > 10 ) // && (avgpxcount < minpxthresh ) && (avgpxcount > 9) ) // && ( circly > 0.91 ) && (circly < 1) && (ffactor > 0.37) && (acc > -2) && (acc < 2) )
			{
				true_det.push_back(det[i]);							// Faraway detections found
			}
			
		}
		else
		{
			
		if( (!(det[i].age < lesser_age)) &&  !( maxpxcount > avgpxthresh ) && (!(det[i].age > large_age)) && (  std::sqrt( (P.x * P.x) + (P.y * P.y) ) < max_movement  ) && (age_ratio  > 0.45) && (aspect_ratio == 1 || sqrt(pow(aspect_ratio - 1,2)) <= 0.5 ) && ( (circly*aspect_ratio > 0.5 )  && (circly*aspect_ratio < 1)  )  )// && (aspect_ratio == 1 || sqrt(pow(aspect_ratio - 1,2)) <= 0.5 ) (  std::sqrt( (P.x * P.x) + (P.y * P.y) ) < max_movement  ) && (age_ratio  > 0.45)  )// && (det[i].age / det[i].total_age  > 0.45) ) //&& (det[i].intime)) && (stddev > 10 ) // && (avgpxcount < minpxthresh ) && (avgpxcount > 9) ) // && ( circly > 0.91 ) && (circly < 1) && (ffactor > 0.37) && (acc > -2) && (acc < 2) )
		{
		true_det.push_back(det[i]);							// Faraway detections found
		}
			
			
		}
		
					//
					
//	po.push_back(det[i]);	
												// All invisible ones saved in --> po <---
				}
				
				else										// All those objects which have some invisible age but havent gone invisible							
				{
				Dnew.push_back(det[i]);	            // 18 //					// Objects which did not go invisible (use them in the next iteration) (SURELY VISIBLE OBJECTS)
				}
		
			}
			else											// All those objects which are currently visible (Zero invisible age)
			{
			Dnew.push_back(det[i]);                      // 18 //
			}
			
		}
													// invisible age interface
   det = Dnew;		// 18 //				// passing on non-invisible objects onto the next iteration
   win_flag++;
   FRAMEID++;	
	}  // End of while loop.
	
/*		
	for(int K = 0; K< det.size();K++)		
	{
	po.push_back(det[K]);	
	}			// For pushing back remaining detections
*/	

	cout<<"\n Initial Processing Complete! Renew Process started!!\n";
	
	true_det = renew_pross(MSFPath,string(sname),true_det,segmentID);		// 19 //	// Reprocessing using edge video obtained from MSF
	
	cout<<"\n Renew Process Complete!\n";
	
	
	of<<"\n 					True Detections					\n";
	of<<"\n Filename::"<<nname<<"\n";
	of<<"\n Segment ID::"<<segmentID<<"\n";                    // 20 //
	
	for(int i = 0; i< true_det.size(); i++)
	{
	
		vector<float> circ_vector;
		int max = 0;
		int avg = 0;
for(int u = 0; u< true_det[i].pxcount.size(); u++){ if(max < true_det[i].pxcount[u] ){max = true_det[i].pxcount[u];} avg+= true_det[i].pxcount[u]; } 
	avg /= true_det[i].pxcount.size();
		
				float rad = 0;
				for(int l = 0; l < true_det[i].minboundradius.size(); l++ )
				{
				
					rad += true_det[i].minboundradius[l];
					
				}
				rad /= true_det[i].minboundradius.size();
		
				int wid = 0;
				int hei = 0;
				for(int l = 0;l< true_det[i].width.size(); l++ )
				{
					wid += true_det[i].width[l];
					hei += true_det[i].height[l];
				}
				float aspect_ratio = float(hei /= true_det[i].height.size()) / float(wid /= true_det[i].width.size() ) ;
				
		double circly;
		
	of<<"\n";
	of<<i+1<<". Start Frame:"<<true_det[i].frameID[0]<<" End Frame:"<<true_det[i].frameID[ true_det[i].frameID.size() - 1 ]<<" Age::  "<<true_det[i].age<<" Total Age::"<<true_det[i].total_age<<" Tracker ID::"<<true_det[i].tID<<"\n";
	of<<"\n Average Pixels::"<<avg<<" Max Pixels::"<<max;	circly = ( (4*avg)/(CV_PI * rad*2*rad*2) );
	of<<"\n Circularity:: "<<circly<<"	FrameID:: [ ";
	for(int l = 0; l < true_det[i].frameID.size(); l++ )	{of<<true_det[i].frameID[l]<<"	";} of<<" ]		\n";
	of<<"\n Average Aspect Ratio:::"<<aspect_ratio; 
				
					// Since FRAMEID is a 0 based index, no need to do a (x - 1)
	 int Y = 0;
	 int p = 0;

	 while(Y < true_det[i].frameID.size() )			// 	 while(Y <= true_det[i].frameID.size() )
	 {
cap.set(CV_CAP_PROP_POS_FRAMES,true_det[i].frameID[Y]);
		 Mat F; cap>>F;

//	    rectangle(F,true_det[i].bb.tl(),true_det[i].bb.br(),Scalar(0,0,255),5,CV_AA); // draw rectangle
	  circle(F,true_det[i].trace[Y],5,Scalar(0,0,255),1);
	 	string temmp = string(nname);
		 stringstream ii;
		 ii<<i;
		 
	 		string imgpath("/media/ml/New Volume1/W_motion/Results_BF_Data/");
//	 		string imgpath2("/home/ml/motion/R_MASK/");
			stringstream convert;
			stringstream seg; seg<< segmentID;
			convert <<true_det[i].frameID[Y];
		 	imgpath.append("dwell");
		 	imgpath.append(ii.str());
		 	imgpath.append("_segmentID_");
		 	imgpath.append(seg.str());
			imgpath.append(temmp.data());
			imgpath.append(convert.str());
			imgpath.append(".jpeg");
			int kswitch = imwrite(imgpath.data(),F);
//			kswitch = imwrite(imgpath2.data(),true_det[i].Mask);
	
	  	 Y++;
	 }
	
	}
	
	
	remove(MSFPath.data());
	remove(morphed.data());                                // 21 //
	of.close();

	

	cout<<"\nVideo Processing Complete!!!\n";
	cap.release();
	clear_memory();
}// end of filepross	

int main()
{
//  1  //
DIR *pdir = NULL;
char dirname[] = "/media/ml/New Volume1/T_W_SP";		// The address need not be added with the customary '/' at the end (Workspace made outside main linux memory to save space)
pdir = opendir(dirname); // ENTER DIRECTORY CONTAINING ALL FILE NAMES HERE
int f_ID = 0;
int file_segments = 6;	
if(pdir ==  NULL)
{
cout<<"\n\n FATAL ERROR!!! FILE NOT READ";
exit(1);
}	

struct dirent *pent = NULL;
// pent = readdir(pdir);
		
while( pent = readdir(pdir) )
{
	
	if(pent == NULL)
	{
	cout<<"\nFILE READING ERROR!!!!!\n";
	return 1;
	}
	
	string * filnam = new string( pent->d_name );
	if(filnam->at(0) == '.') { continue; }
	
	cout<<"\n Processing File Name: "<<pent->d_name<<"\n\n";
	
	char sname[100];
	sprintf(sname,"%s/%s",dirname,pent->d_name);
//	regis_video(string(sname));

	vector<string> alist;
	stabilize_video(string(sname));                                 // 2 //                       // in the final implementation THIS HAS TO BE ACTIVATED BEFORE USE
	create_video_files(alist, string(sname));			// 3 //       // (Already done for these files)
	
	clock_t cl1 = clock();
	
        for(int i = 0; i< alist.size(); i++)
        {
        filepross(alist[i].data(),i,pent->d_name,i);	               // 4 //
        }
        clock_t cl2 = clock();		
	
	cout<<"\n Time taken for processing one file:::"<<(cl2 - cl1)/CLOCKS_PER_SEC<<"\n";
	
	delete_video_files(alist);                                     // 23 //
	f_ID++;

}
	
	return 1;
}                                                                       // 24 //
