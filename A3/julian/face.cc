#include "face.h"

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

/// Start the training.  This resets/initializes the model.
void FACE::startTraining() {
	
	pimpl->allPos.clear();
	pimpl->allNeg.clear();
	pimpl->pts.clear();
	
	for (int i=0; i<DESCRIPTOR_SIZE; i++)
		pimpl->pts.push_back({{rand()%250,rand()%250},{rand()%250,rand()%250}});	
}

/// Add a new person.
///
/// @param img1:  250x250 pixel image containing a scaled and aligned face
/// @param img2:  250x250 pixel image containing a scaled and aligned face
/// @param same: true if img1 and img2 belong to the same person
void FACE::train(const cv::Mat3b& img1, const cv::Mat3b& img2, bool same) {
	
	std::vector<double> desc;
	for (auto &p : pimpl->pts)
		desc.push_back( ((img1(p.first)[1]<img1(p.second)[1]) == (img2(p.first)[1]<img2(p.second)[1])) - .5);
		
	if (same)
		pimpl->allPos.push_back(desc);
	else 
		pimpl->allNeg.push_back(desc);	
}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
void FACE::finishTraining() {
	
	pimpl->pos = vector<double>(DESCRIPTOR_SIZE,0);
	for (auto &d : pimpl->allPos)
		for (uint i=0; i<d.size(); i++) 
			pimpl->pos[i] += d[i]/pimpl->allPos.size();

	pimpl->neg = vector<double>(DESCRIPTOR_SIZE,0);
	for (auto &d : pimpl->allNeg)
		for (uint i=0; i<d.size(); i++) 
			pimpl->neg[i] += d[i]/pimpl->allNeg.size();
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

	std::vector<double> desc;
	for (auto &p : pimpl->pts)
		desc.push_back( ((img1(p.first)[1]<img1(p.second)[1]) == (img2(p.first)[1]<img2(p.second)[1])) - .5);

	double scorePos=0, scoreNeg=0;
	for (uint i=0; i<desc.size(); i++) {
		scorePos += pimpl->pos[i]*desc[i];
		scoreNeg += pimpl->neg[i]*desc[i];
	}

	return scorePos/(scoreNeg+1E-10);
}

