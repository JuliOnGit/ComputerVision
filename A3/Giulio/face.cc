#include "face.h"

#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

static const int DESCRIPTOR_SIZE = 25600;

struct FACE::FACEPimpl {
	
	std::vector<double> pos, neg;
	std::vector<std::vector<double>> allPos, allNeg;
	std::vector<std::pair<cv::Point2i,cv::Point2i>> pts;
};


/// Constructor
FACE::FACE() : pimpl(new FACEPimpl()) {
}

/// Destructor
FACE::~FACE() {
}

cv::Ptr<cv::face::EigenFaceRecognizer> eigenModel;
cv::Ptr<cv::face::LBPHFaceRecognizer> histogramModel;

std::vector<cv::Mat> faces;

/// Start the training.  This resets/initializes the model.
void FACE::startTraining() {
	
	/*pimpl->allPos.clear();
	pimpl->allNeg.clear();
	pimpl->pts.clear();
	
	for (int i=0; i<DESCRIPTOR_SIZE; i++)
		pimpl->pts.push_back({{rand()%250,rand()%250},{rand()%250,rand()%250}});
	*/	

	eigenModel = cv::face::EigenFaceRecognizer::create();
	histogramModel = cv::face::LBPHFaceRecognizer::create(1, 8, 12, 12);
}

cv::Mat adjustImage(const cv::Mat3b& img) {

	cv::Mat grey;
	cv::cvtColor(img, grey, cv::COLOR_BGR2GRAY);
	cv::Mat cropped = grey(cv::Rect(75, 50, 100, 150));

	return cropped;
}

cv::Mat getHistogram(const cv::Mat1b& img) {

	std::vector<cv::Mat> face;
	std::vector<int> labels;
	
	labels.push_back(0);
	face.push_back(img);
	
	histogramModel->train(face, labels);

	return histogramModel->getHistograms()[0];
}


/// Add a new person.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @param same: true if img1 and img2 belong to the same person
void FACE::train(const cv::Mat3b& img1, const cv::Mat3b& img2, bool same) {
	
	/*std::vector<double> desc;
	for (auto &p : pimpl->pts)
		desc.push_back( ((img1(p.first)[1]<img1(p.second)[1]) == (img2(p.first)[1]<img2(p.second)[1])) - .5);
		
	if (same)
		pimpl->allPos.push_back(desc);
	else 
		pimpl->allNeg.push_back(desc);
	*/

	cv::Mat cropped1 = adjustImage(img1);
	cv::Mat cropped2 = adjustImage(img2);

	faces.push_back(cropped1);
	faces.push_back(cropped2);

	cv::Mat histogram1 = getHistogram(cropped1);
	cv::Mat histogram2 = getHistogram(cropped2);

	double dist = cv::norm(histogram1, histogram2, cv::NORM_L2);

}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void FACE::finishTraining() {
	
	/*pimpl->pos = vector<double>(DESCRIPTOR_SIZE,0);
	for (auto &d : pimpl->allPos)
		for (uint i=0; i<d.size(); i++) 
			pimpl->pos[i] += d[i]/pimpl->allPos.size();

	pimpl->neg = vector<double>(DESCRIPTOR_SIZE,0);
	for (auto &d : pimpl->allNeg)
		for (uint i=0; i<d.size(); i++) 
			pimpl->neg[i] += d[i]/pimpl->allNeg.size();
	*/

	std::vector<int> labels(faces.size());
	eigenModel->train(faces, labels);
	
	//eigenModel->read("eigen.model");

}

/// Verify if img corresponds to the provided name.  The result is a floating point
/// value directly proportional to the probability of being correct.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @return:    similarity score between both images
double FACE::verify(const cv::Mat3b& img1, const cv::Mat3b& img2) {
	
//	return rand()%256;
//	return -cv::norm(img1-img2);

	/*std::vector<double> desc;
	for (auto &p : pimpl->pts)
		desc.push_back( ((img1(p.first)[1]<img1(p.second)[1]) == (img2(p.first)[1]<img2(p.second)[1])) - .5);

	double scorePos=0, scoreNeg=0;
	for (uint i=0; i<desc.size(); i++) {
		scorePos += pimpl->pos[i]*desc[i];
		scoreNeg += pimpl->neg[i]*desc[i];
	}

	return scorePos/(scoreNeg+1E-10);*/	

	cv::Ptr<cv::face::EigenFaceRecognizer> eigenModel1 = cv::face::EigenFaceRecognizer::create();
	cv::Ptr<cv::face::EigenFaceRecognizer> eigenModel2 = cv::face::EigenFaceRecognizer::create();

	std::vector<cv::Mat> face1, face2; 
	std::vector<int> label(1);

	cv::Mat cropped1 = adjustImage(img1);
	cv::Mat cropped2 = adjustImage(img2);

	face1.push_back(cropped1);
	face2.push_back(cropped2);

	eigenModel1->train(face1, label);
	eigenModel2->train(face2, label);

	cv::Mat U = eigenModel->getEigenVectors();
	cv::Mat m = eigenModel->getMean();
	cv::Mat y1 = eigenModel1->getMean();
	cv::Mat y2 = eigenModel2->getMean();

	cv::Mat res1 = U.t() * (y1-m).t();
	cv::Mat res2 = U.t() * (y2-m).t();

	cv::Mat histogram1 = getHistogram(cropped1);
	cv::Mat histogram2 = getHistogram(cropped2);

	double dist = 1000 * cv::norm(histogram1, histogram2, cv::NORM_L2) + cv::norm(res1, res2, cv::NORM_L2);

	return -dist;
}