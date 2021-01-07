#include "skinmodel.h"
#include <cmath>
#include <iostream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <opencv2/ml/ml.hpp>

#include <map>

using namespace std;

bool useMLP = true;


/// Constructor
SkinModel::SkinModel()
{
}

/// Destructor
SkinModel::~SkinModel() 
{
}


struct GaussianModel {
	cv::Vec3f mean = cv::Vec3f();
	cv::Mat1f covariance = cv::Mat1f::zeros(3, 3);
} skinModel, nonSkinModel;

int trainingCount;

cv::Vec3f convertVec3dToVec3f(const cv::Vec3d vec) {
	return cv::Vec3f(vec[0], vec[1], vec[2]);
} 

cv::Mat3b normalizeImage(const cv::Mat3b& img, const std::string f) {

	cv::Mat3b normalizedImage = cv::Mat3b::zeros(img.rows, img.cols);

	cv::cvtColor(img, normalizedImage, cv::COLOR_BGR2YCrCb);

	vector<cv::Mat> channels;

	cv::split(normalizedImage, channels);

	cv::equalizeHist(channels[0], channels[0]);

	cv::merge(channels, normalizedImage);

	cv::cvtColor(normalizedImage, normalizedImage, cv::COLOR_YCrCb2BGR);

	if (f != "") {
		boost::filesystem::create_directory(boost::filesystem::path("out/normalize"));
		cv::imwrite("out/normalize/out-"+f,normalizedImage);
	}

	return normalizedImage;
} 

cv::Mat1b filterOutputImage(const cv::Mat1b img, const std::string f, int erosionSize = 2, int dilateSize = 2) {

	cv::Mat1b filteredSkin = cv::Mat1b::zeros(img.rows, img.cols);

	cv::Mat erodeElement = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2*erosionSize+1, 2*erosionSize+1), cv::Point(erosionSize,erosionSize));
	cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(2*dilateSize+1, 2*dilateSize+1), cv::Point(dilateSize,dilateSize));

	cv::dilate(img, filteredSkin, dilateElement);
	cv::erode(filteredSkin, filteredSkin, erodeElement);

	cv::erode(filteredSkin, filteredSkin, erodeElement);
	cv::dilate(filteredSkin, filteredSkin, dilateElement);
	

	if (f != "") {
		boost::filesystem::create_directory(boost::filesystem::path("out/filter"));
		cv::imwrite("out/filter/out-"+f,filteredSkin);
	}
	

	return filteredSkin;

}


int hiddenLayerSize = 25;


cv::Ptr<cv::ml::ANN_MLP> mlp;


/// Start the training.  This resets/initializes the model.
///
/// Implementation hint:
/// Use this function to initialize/clear data structures used for training the skin model.
void SkinModel::startTraining()
{
    //--- IMPLEMENT THIS ---//

	trainingCount = 0;

	mlp = cv::ml::ANN_MLP::create();

	cv::Mat layerSize = cv::Mat(3, 1, CV_16U);
	layerSize.row(0) = 3;
	layerSize.row(1) = hiddenLayerSize;
	layerSize.row(2) = 1;

	mlp->setLayerSizes(layerSize);
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0, 0);
	
	cv::TermCriteria termCrit = cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS, 100, 0.000000001);
	mlp->setTermCriteria(termCrit);

	mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.000000001);

}



/// Add a new training image/mask pair.  The mask should
/// denote the pixels in the training image that are of skin color.
///
/// @param img:  input image
/// @param mask: mask which specifies, which pixels are skin/non-skin
void SkinModel::train(const cv::Mat3b& img, const cv::Mat1b& mask)
{
	//--- IMPLEMENT THIS ---//

	cv::Mat3b normalizedImage = img; //normalizeImage(img, "");

	if (!useMLP) {

		struct GaussianModel iterationSkinModel, iterationNonSkinModel;

		int skinPixelCount = 0;
		int nonSkinPixelCount = 0;

		for (int row = 0; row < img.rows; row++) {
			for (int col = 0; col < img.cols; col++) {

				cv::Vec3f pixel = convertVec3dToVec3f(normalizedImage(row, col));

				if (mask(row, col) != 0) {
					// skin
					iterationSkinModel.mean += pixel;
					skinPixelCount++;

				} else {
					// no skin
					iterationNonSkinModel.mean += pixel;
					nonSkinPixelCount++;
				}

			}
		}

		iterationSkinModel.mean /= skinPixelCount;
		iterationNonSkinModel.mean /= nonSkinPixelCount;

		skinModel.mean += iterationSkinModel.mean;
		nonSkinModel.mean += iterationNonSkinModel.mean;

		for (int row = 0; row < img.rows; row++) {
			for (int col = 0; col < img.cols; col++) {

				cv::Vec3f pixel = convertVec3dToVec3f(normalizedImage(row, col));

				if (mask(row, col) != 0) {
					// skin
					cv::Vec3f diff = pixel - iterationSkinModel.mean;
					iterationSkinModel.covariance += (diff * diff.t());
				} else {
					// no skin
					cv::Vec3f diff = pixel - iterationNonSkinModel.mean;
					iterationNonSkinModel.covariance += (diff * diff.t());
				}

			}
		}

		skinModel.covariance += iterationSkinModel.covariance / (skinPixelCount-1);
		nonSkinModel.covariance += iterationNonSkinModel.covariance / (nonSkinPixelCount-1);

	}

	if (useMLP) {

		cv::Mat inputTrainingData = cv::Mat(img.rows*img.cols, 3, CV_32F);
		cv::Mat outputTrainingData = cv::Mat(img.rows*img.cols, 1, CV_32F);

		for (int row = 0; row < img.rows; row++) {
			for (int col = 0; col < img.cols; col++) {

				cv::Vec3f pixel = convertVec3dToVec3f(normalizedImage(row, col));

				inputTrainingData.at<float>(row + col * img.rows, 0) = pixel[0];
				inputTrainingData.at<float>(row + col * img.rows, 1) = pixel[1];
				inputTrainingData.at<float>(row + col * img.rows, 2) = pixel[2];


				if (mask(row, col) != 0) {
					// skin

					outputTrainingData.at<float>(row + col * img.rows, 0) = 1;

				} else {
					// no skin

					outputTrainingData.at<float>(row + col * img.rows, 0) = 0;

				}

			}
		}
		

		cv::Ptr<cv::ml::TrainData> trainingData = cv::ml::TrainData::create(inputTrainingData, cv::ml::SampleTypes::ROW_SAMPLE, outputTrainingData);
		
		if (trainingCount == 0) {
			mlp->train(trainingData, cv::ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE | cv::ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);
		} else {
			mlp->train(trainingData, cv::ml::ANN_MLP::TrainFlags::UPDATE_WEIGHTS | cv::ml::ANN_MLP::TrainFlags::NO_INPUT_SCALE | cv::ml::ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);
		}

	}
	
	trainingCount++;

}

/// Finish the training.  This finalizes the model.  Do not call
/// train() afterwards anymore.
///
/// Implementation hint:
/// e.g normalize w.r.t. the number of training images etc.
void SkinModel::finishTraining() {

	//--- IMPLEMENT THIS ---//

	if (!useMLP) {
		
		skinModel.mean /= trainingCount;
		nonSkinModel.mean /= trainingCount;

		skinModel.covariance /= trainingCount;
		nonSkinModel.covariance /= trainingCount;

		cv::invert(skinModel.covariance, skinModel.covariance, cv::DECOMP_LU);
		cv::invert(nonSkinModel.covariance, nonSkinModel.covariance, cv::DECOMP_LU);
	}

	if (useMLP) {
		//inputTrainingData = cv::Mat(trainingCount, 1);
	}
}

float calculateProbability(const cv::Vec3f& pixel, const GaussianModel& model) {

	cv::Vec3f difference = pixel - model.mean;
	
	cv::Mat1f v = difference.t() * model.covariance;

	cv::Mat1f v2 = v * difference;

	float exponent = -(1.f/2) * v2[0][0];

	float scale = powf(2*M_PI, -3.f/2) * powf(cv::determinant(model.covariance), -1.f/2);

	return scale * exp(exponent);

}


/// Classify an unknown test image.  The result is a probability
/// mask denoting for each pixel how likely it is of skin color.
///
/// @param img: unknown test image
/// @return:    probability mask of skin color likelihood
cv::Mat1b SkinModel::classify(const cv::Mat3b& img, std::string f)
{
    cv::Mat1b skin = cv::Mat1b::zeros(img.rows, img.cols);

	//--- IMPLEMENT THIS ---//
	
	cv::Mat3b normalizedImage = img; //normalizeImage(img, "");

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {

			cv::Vec3f pixel = convertVec3dToVec3f(normalizedImage(row, col));

			if (useMLP) {

				cv::Mat sample = cv::Mat(1, 3, CV_32F);
				sample.at<float>(0, 0) = pixel[0];
				sample.at<float>(0, 1) = pixel[1];
				sample.at<float>(0, 2) = pixel[2];

				cv::Mat res;
				
				mlp->predict(sample, res);

				if (res.at<float>(0,0) > 0.5f) {
					skin(row, col) = 255;
				}
			}


			if (!useMLP) {
				float t = 0.24f;

				if (calculateProbability(pixel, skinModel) >= t * calculateProbability(pixel, nonSkinModel)) {
					skin(row, col) = 255;
				}
			}

			if (false)
				skin(row, col) = rand()%256;

			if (false)
				skin(row, col) = img(row,col)[2];

			if (false) {
			
				cv::Vec3b bgr = img(row, col);
				if (bgr[2] > bgr[1] && bgr[1] > bgr[0]) 
					skin(row, col) = 2*(bgr[2] - bgr[0]);
			}
        }
    }

	
    return filterOutputImage(skin, "");
}

