#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
using namespace cv;
using namespace std;
class Kalman
{
public:
	Mat output;
	Mat input;						
	int stateNum;					
	int measureNum;					
	void set_Q(float x);
	void set_R(float y);
	void set_P(float z);
	void get_F(Mat FF);
	void init();					
	void update();					

private:
	Mat Q;			
	Mat R;			
	Mat F;			
	Mat H;          
	Mat K;			

	Mat P;			
	Mat P_predict;  
	Mat x_hat_prect;
	Mat temp;
};

void::Kalman::set_Q(float x)
{
	Q = x * Mat::eye(stateNum, stateNum, CV_32F);
}

void::Kalman::set_R(float y)
{

	R = y * Mat::eye(measureNum, measureNum, CV_32F);

}


void::Kalman::set_P(float z)
{

	P_predict = z * Mat::eye(stateNum, stateNum, CV_32F);

}

void::Kalman::get_F(Mat FF)
{
	F = FF.clone();
}


void Kalman::init()
{
	stateNum = 4;
	measureNum = 2;
	K = Mat::zeros(stateNum, stateNum, CV_32F);
	H = Mat::zeros(measureNum, stateNum, CV_32F);
	temp = Mat::zeros(stateNum, stateNum, CV_32F);

	for (int i = 0; i < measureNum; i++)
	{
		H.at<float>(i, i) = 1;
	}
	set_Q(0.0001);
	set_R(0.01);
	set_P(1);
	output = (Mat_<float>(4, 1) << 100, 100, 0, 0);

	get_F((Mat_<float>(4, 4) << 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1));
}

void::Kalman::update()
{

	x_hat_prect = F * output;
	P = F * P_predict*F.t() + Q;

	temp = H * P*H.t() + R;
	temp = temp.inv();
	K = P * H.t() *temp;

	output = x_hat_prect + K * (input - H * x_hat_prect);     				
	P_predict = (Mat::eye(stateNum, stateNum, CV_32F) - K * H)*P;		

}

Point update() 
{
	static int i = 0;
	Point res = Point(250 - 100 * cos(i*0.1), 250 - 100 * sin(i*0.1));
	i++;
	if (i == 36000)i = 0;
	return res;
}
Kalman kalman1;
Kalman kalman2;
void  process1(Mat img) 
{
	cvtColor(img, img, COLOR_BGR2GRAY);
	threshold(img, img, 100, 255,0);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  //只找最外层轮廓

	cv::Mat dst = Mat::zeros(img.size(),CV_8UC3);
	Rect rect;
	int tag = 0;
	for (int i = 0; i < contours.size(); ++i)
	{  //绘制所有轮廓
		if (contours[i].size() > 50) 
		{
			cv::drawContours(dst, contours, i, cv::Scalar(255, 255, 255), -1);  //thickness为-1时为填充整个轮廓
			rect = boundingRect(contours[i]);
			rectangle(dst, rect, Scalar(0, 0, 255), 1);
			tag = 1;
		}
	}
	Point center = Point(rect.x+rect.width/2, rect.y+rect.height/2);
	
	Point pos=center;
	if (tag==0) 
	{
			pos.x = kalman1.output.at<float>(0) + kalman1.output.at<float>(2);
			pos.y = kalman1.output.at<float>(1) + kalman1.output.at<float>(3);
			circle(img, pos, 10, Scalar(0, 255, 255), -1);
		}
		else
			circle(img, pos, 10, Scalar(0, 0, 255), -1);
	circle(dst, pos, 3, (0, 0, 255), -1);
	circle(dst, pos, 20, (0, 0, 255), 10);
	kalman1.input = (Mat_<float>(2, 1) << pos.x, pos.y);     						//测量值
	kalman1.update();
	imshow("dst", dst);
}

int dir1;
int dir2;
Point pos1;
Point pos2;
Rect rect1;
Rect rect2;
Point center1;
Point center2;
int cc = 0;
double iou(Rect rect, Rect rect1) 
{
	Rect rect2 = rect | rect1;
	//cout << rect2.x << ";" << rect2.y << ";" << rect2.width << ";" << rect2.height << ";" << rect2.area() << endl;

	//计算两个举行的并集
	Rect rect3 = rect & rect1;
	//cout << rect3.x << ";" << rect3.y << ";" << rect3.width << ";" << rect3.height << ";" << rect3.area() << endl;

	//计算IOU
	double IOU = rect3.area() *1.0 / rect2.area();
	return IOU;
}

void  process2(Mat img) 
{

	cvtColor(img, img, COLOR_BGR2GRAY);
	threshold(img, img, 100, 255, 0);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  //只找最外层轮廓

	cv::Mat dst = Mat::zeros(img.size(), CV_8UC3);
	Rect rect[10];
	int rect_len = 0;
	for (int i = 0; i < contours.size(); ++i)
	{  //绘制所有轮廓
		if (contours[i].size() > 50) 
		{
			cv::drawContours(dst, contours, i, cv::Scalar(255, 255, 255), -1);  //thickness为-1时为填充整个轮廓
			rect[rect_len++] = boundingRect(contours[i]);
		}
	}
	Rect rect3;
	Rect rect4;
	if (cc > 2) 
	{
		pos1.x = kalman1.output.at<float>(0) + kalman1.output.at<float>(2);
		pos1.y = kalman1.output.at<float>(1) + kalman1.output.at<float>(3);
		pos2.x = kalman2.output.at<float>(0) + kalman2.output.at<float>(2);
		pos2.y = kalman2.output.at<float>(1) + kalman2.output.at<float>(3);
		rect3 = rect1;
		rect3.x = pos1.x - rect1.width / 2;
		rect3.y = pos1.y - rect1.height / 2;
		rect4 = rect2;
		rect4.x = pos2.x - rect2.width / 2;
		rect4.y = pos2.y - rect2.height / 2;
	}
	if (rect_len == 1) 
	{
		rect1 = rect[0];
		rect2 = rect[1];
	}
	else 
	{
		
		double rate1 = iou(rect1, rect[0]);
		double rate2 = iou(rect1, rect[1]);
		if (cc > 2) 
		{
			rate1 = iou(rect3, rect[0]);
			rate2 = iou(rect3, rect[1]);
		}
		if (rate1 > rate2) 
		{
			rect1 = rect[0];
			rect2 = rect[1];
		}
		else 
		{
			rect1 = rect[1];
			rect2 = rect[0];
		}
	}
	center1 = Point(rect1.x + rect1.width / 2, rect1.y + rect1.height / 2);
	center2 = Point(rect2.x + rect2.width / 2, rect2.y + rect2.height / 2);
	kalman1.input = (Mat_<float>(2, 1) << center1.x, center1.y);     
	kalman2.input = (Mat_<float>(2, 1) << center2.x, center2.y); 
	kalman1.update();
	kalman2.update();
	rectangle(dst, rect1, Scalar(0, 0, 255), 1);
	rectangle(dst, rect2, Scalar(255, 0, 0), 1);
	imshow("dst", dst);
	cc++;
}

int test1() 
{
	kalman1.init();
	namedWindow("name");
	VideoCapture cap("1.avi");//读取视频文件

	Mat frame;
	while (true) 
	{

		cap.read(frame);//frame为输出，read是将捕获到的视频一帧一帧的传入frame

		//对视频读取时，同图像一样会有判空操作
		if (frame.empty()) 
		{
			break;
		}

		imshow("frame", frame);
		process1(frame);
		int c = waitKey(50);
		if (c == 50) 
		{
			break;
		}
	}
	cap.release();
	return 0;
}

int test2() 
{
	kalman1.init();
	kalman2.init();
	namedWindow("name");
	VideoCapture cap("2.avi");//读取视频文件

	Mat frame;
	while (true) 
	{

		cap.read(frame);//frame为输出，read是将捕获到的视频一帧一帧的传入frame

		//对视频读取时，同图像一样会有判空操作
		if (frame.empty())
		{
			break;
		}

		imshow("frame", frame);
		process2(frame);
		int c = waitKey(500);
		if (c == 50)
		{
			break;
		}
	}
	cap.release();
	return 0;
}

int main() 
{
	test1();
	test2();
	return 0;
}