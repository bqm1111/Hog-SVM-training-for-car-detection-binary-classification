#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include<stdio.h>
#include<windows.h>

#define WINDOW_NAME "WINDOW"
#define TRAFFIC_VIDEO_FILE "F:\\COMPUTER VISION\\dataset\\VIDEO\\GROUND\\13.mkv"
#define TRAINED_SVM "vehicle_detector.yml"
#define	IMAGE_SIZE Size(72, 32) 
#define JACCARD_INDEX 0.8

using namespace cv;
using namespace cv::ml;
using namespace std;

void load_images(string directory, vector<Mat>&image_lst);			
void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);// chuyen du lieu tu dang vector ve matrix
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size);  // extract hog feature
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels);			// huan luyen su dung svm
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);		// ve bounding box dua tren doi tuong phat hien duoc
void test(const Size & size);																
void read_ground_truth(string & dir,vector<vector<Rect>> &a);								// read ground truth from training image
void convert(string &str, vector<Rect> &v, Rect & x);										// chuan hoa string de lay toa do trong training image(xem file README trong folder Cardata de biet them chi tiet)
void recognition_evaluation(vector<Rect>& location,vector<Rect> &ground_truth, int& TP, int&FP, int&FN);	// danh gia do chinh xac cua chuong trinh( tinh recall and precision)

//------------------------------------------------------------------------
int main()
{
	vector< Mat > pos_lst;
	vector< Mat > full_neg_lst;
	vector< Mat > neg_lst;
	vector< Mat > gradient_lst;
	vector< int > labels;
	// positive file
		load_images("F:\\Driver E\\CarData\\positive\\",pos_lst);
		labels.assign(pos_lst.size(), +1);
		//negative file
		load_images("F:\\Driver E\\CarData\\negative\\", full_neg_lst);
		labels.insert(labels.end(), full_neg_lst.size(), -1);
		for (int j = 0; j < labels.size(); j++)
			cout << labels[j] << endl;
		compute_hog(pos_lst, gradient_lst, IMAGE_SIZE);
		compute_hog(full_neg_lst, gradient_lst, IMAGE_SIZE);

		train_svm(gradient_lst, labels);
		//test video
		test(IMAGE_SIZE);
		return 0;
	}


//-------------------------------------load image-------------------------------------------------

void load_images(string directory, vector<Mat>& image_list) {

	Mat img;
	string dir = directory + "/*";
	static const char* chFolderpath = dir.c_str();  
	string data;
	HANDLE hFind;
	WIN32_FIND_DATAA data2;
	int i = 0;
	hFind = FindFirstFile(chFolderpath, &data2);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			data = "";
			i++;
			if (i < 3) continue;//( phai bo qua 2 gia tri dau vi 2 gia tri dau la duong dan toi thu muc va thu muc me cua no. ket qua cua data2.cFileName se la"." va"..")
			data += directory;
			data += data2.cFileName;
			cout << data << endl;
			img = imread(data);
			imshow(WINDOW_NAME,img);
			if (img.empty())
				continue;
			resize(img, img, IMAGE_SIZE);
			image_list.push_back(img.clone());
		} while (FindNextFile(hFind, &data2));
		FindClose(hFind);
	}
}


//--------------------------------get support vector------------------------------------------

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);// change 0 into N(N-1)/2 in N-class classification problem (N>2)

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
}

//----------------------------convert training data from vector datatype to Mat datatype------------------------------------------------------------------------------
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = train_samples[0].cols> train_samples[0].rows? train_samples[0].cols: train_samples[0].rows;
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);
	vector< Mat >::const_iterator itr = train_samples.begin();
	vector< Mat >::const_iterator end = train_samples.end();
	for (int i = 0; itr != end; ++itr, ++i)
	{
		CV_Assert(itr->cols == 1 ||
			itr->rows == 1);
		if (itr->cols == 1)
		{
			transpose(*(itr), tmp);
			tmp.copyTo(trainData.row(i));
		}
		else if (itr->rows == 1)
		{
			itr->copyTo(trainData.row(i));
		}
	}
}
//------------------------------extract HOG feature-------------------------------------------------------------
void compute_hog(const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size)
{
	HOGDescriptor hog;
	hog.winSize = size;
	hog.blockSize = Size(16, 16);
	hog.blockStride = Size(8, 8);
	hog.cellSize = Size(8, 8);
	Mat gray;
	vector< Point > location;
	vector< float > descriptors;

	vector< Mat >::const_iterator img = img_lst.begin();
	vector< Mat >::const_iterator end = img_lst.end();
	for (; img != end; ++img)
	{
		cvtColor(*img, gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(72, 32), Size(0, 0), location);
		gradient_lst.push_back(Mat(descriptors).clone());
	}
}
//--------------------------------------------------------------------------------------------------------------------
void train_svm(const vector< Mat > & gradient_lst, const vector< int > & labels)
{
	/* set parameter */
	Ptr<SVM> svm = SVM::create();
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);
	svm->setNu(0.5);
	svm->setP(0.1); 
	svm->setC(0.01); 
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; 

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);

	clog << "Start training...";
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	svm->save(TRAINED_SVM);
}
//---------------------------------draw location--------------------------------------------------------------------------------
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		}
	}
}
//----------------------------------------------test the result-------------------------------------------------------
//Note: press n to see the next picture
//      press b to see the previous picture
//		to skip the picture results in order to the final results of precision and recall: press esc
void test(const Size & size)
{
	int N=170;
	char key=0;
	int i = 0;
	int TP=0, FN=0, FP=0;
	Mat img, draw;
	Ptr<SVM> svm;
	HOGDescriptor hog;
	hog.winSize = size;
	hog.blockSize = Size(16, 16);
	hog.blockStride = Size(8, 8);
	hog.cellSize = Size(8, 8);
	vector< Rect > locations;
	vector<vector<Rect>> a;
	string dir = "F:\\Driver E\\CarData\\CarData\\trueLocations.txt";
	read_ground_truth(dir, a);
	char FirstFileName[100] = "F:\\Driver E\\CarData\\CarData\\TestImages\\test-";//"F:\\Driver E\\CarData\\CarData\\TestImages\\test-";
	char FullFileName[100];
	// Load the trained SVM.
	svm = StatModel::load<SVM>(TRAINED_SVM);
	// Set the trained svm to my_hog
	vector< float > hog_detector;
	get_svm_detector(svm, hog_detector);
	hog.setSVMDetector(hog_detector);
	for (int i = 0; i < N; i++)
	{
		sprintf_s(FullFileName, "%s%d.pgm", FirstFileName, i);
		img = imread(FullFileName);
		bool end_of_process = false;
		if (img.empty())
			break;
		draw = img.clone();
		locations.clear();
		hog.detectMultiScale(img, locations);
		recognition_evaluation(locations, a[i], TP, FP, FN);
		cout << "TP=" << TP << endl;
		cout << "FP=" << FP << endl;
		cout << "FN=" << FN << endl;
		while (!end_of_process&&key!=27)
		{
			if (img.empty())
				break;
			draw = img.clone();
			locations.clear();
			hog.detectMultiScale(img, locations);// to find location in img)
			draw_locations(draw, locations, Scalar(0, 255, 0));
			imshow(WINDOW_NAME, draw);
			
			key = (char)waitKey(0);
			if ('n' == key)
				end_of_process = true;
			else if(key=='b')
			{
				i=i-2;
				end_of_process = true;
			}
		}
		/*video.open("F:\\COMPUTER VISION\\dataset\\VIDEO\\GROUND\\9.mp4");
		if (!video.isOpened())
		{
			cerr << "Unable to open the device" << endl;
			exit(-1);
		}


		bool end_of_process = false;
		while (!end_of_process)
		{
			video >> img;
			if (img.empty())
				break;

			draw = img.clone();

			locations.clear();
			hog.detectMultiScale(img, locations);
			draw_locations(draw, locations, Scalar(0, 255, 0));
			imshow(WINDOW_NAME, draw);
			key = (char)waitKey(33);
			if (27 == key)
				end_of_process = true;
		}*/
	}
	cout << "recall= " << (float)TP / (float)(TP + FN) << endl;
	cout << "precision=" << (float)TP / (float)(TP + FP) << endl;
	system("pause");
}
//----------------------------------------------------------------------------
void read_ground_truth(string& dir, vector<vector<Rect>> &a){
	fstream f;
	f.open(dir, ios::in);  // open file"F:\\Driver E\\CarData\\CarData\\trueLocations.txt"
	int num = 0;
	vector<Rect> b;
	Rect c;
	string line; //data in 1 line
	string data;
	while (!f.eof())
	{
		b.clear();
		getline(f, line);
		int i, j;
		// eliminate numerical orders
		for (i = 0; i < line.length(); i++)
		{

			if (line[i] == '(')
			{
				line = line.substr(i); break;
			}
		}
		//count number of space in line
		for (i = 0; i < line.length(); i++)
		{
			if (line[i] == ' ') num++;
		}
		//convert line into vector
		if (num == 0)
		{
			convert(line, b, c);
		}
		else {
			for (i = 0; i < line.length(); i++)
			{
				if (line[i] == ' ')
				{
					data = line.substr(0, i);
					line = line.substr(i + 1);
					convert(data, b, c);
					i = 0;
					num--;//after num==0 still need to convert the last string after the last space
					if (num == 0) {
						convert(line, b, c);
					}
				}
			}
		}
		a.insert(a.end(), 1, b);
	}
	f.close();
	for (int i = 0; i < a.size(); i++) {
		for (int j = 0; j < a[i].size(); j++) {
			cout << a[i][j] << endl;
		}
	}
}
//-----------------------------------------------------------------------------------------
void convert(string &str, vector<Rect> &v, Rect & x)
{
	if (str.length() < 2) return;
	string data = str.substr(1, str.length() - 2);

	for (int j = 0; j < data.length(); j++)
	{
		if (data[j] == ',')
		{
			string temp = data.substr(0, j);
			string temp1 = data.substr(j + 1);
			x = Rect(atoi(temp1.c_str()), atoi(temp.c_str()), 100, 40);
			v.insert(v.end(), 1, x);
			break;
		}
	}
}
//----------------------------------------------------------------------------------------
void recognition_evaluation(vector<Rect>& location, vector<Rect> & ground_truth,int& TP, int&FP,int&FN) {
	// tinh toan dua tren Jaccard index
	int num_loc, num_gt;
	for (int i = 0; i < location.size(); i++) {
		cout << "loc" << "[" << i << "]=" << location[i] << endl;
	}
	int count = 0, j;
	for (num_loc=0; num_loc<location.size(); ++num_loc)
	{
		for (num_gt = 0; num_gt < ground_truth.size();++num_gt)
		{
			cout << "gt=" << ground_truth[num_gt] << endl;
			cout << "loc=" << location[num_loc] << endl;
			float x = (float)((ground_truth[num_gt])&(location[num_loc])).area() / (float)((ground_truth[num_gt]) | (location[num_loc])).area(); //JACARD INDEX FUNCTION
			if (x > JACCARD_INDEX) count++;
			cout << "x=" << x << endl;
		}
	}
	for (num_gt = 0; num_gt < ground_truth.size(); ++num_gt) {
		bool state = false;
		for (num_loc = 0; num_loc<location.size(); ++num_loc) {
			float x = (float)((ground_truth[num_gt])&(location[num_loc])).area() / (float)((ground_truth[num_gt]) | (location[num_loc])).area();
			if (x > JACCARD_INDEX) state=true;
		}
		if (state == false) FN++;
	}
	TP += count;
	FP += ((int)location.size() - count);
}