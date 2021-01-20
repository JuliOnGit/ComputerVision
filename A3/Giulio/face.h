#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <memory>


class FACE {
    struct FACEPimpl;
    std::unique_ptr<FACEPimpl> pimpl;
public:
    /// Constructor
    FACE();

    /// Destructor
    ~FACE();

    /// Start the training.  This resets/initializes the model.
    void startTraining();

    /// Add a new person.
    ///
	/// @param img1:  250x250 pixel image containing a scaled and aligned face
	/// @param img2:  250x250 pixel image containing a scaled and aligned face
	/// @param same: true if img1 and img2 belong to the same person
	void train(const cv::Mat3b& img1, const cv::Mat3b& img2, bool same) ;

    /// Finish the training.  This finalizes the model.  Do not call
    /// train() afterwards anymore.
    void finishTraining();

    /// Verify if img corresponds to the provided name.  The result is a floating point
    /// value directly proportional to the probability of being correct.
    ///
	/// @param img1:  250x250 pixel image containing a scaled and aligned face
	/// @param img2:  250x250 pixel image containing a scaled and aligned face
	/// @return:    similarity score between both images
	double verify(const cv::Mat3b& img1, const cv::Mat3b& img2);

private:
};
