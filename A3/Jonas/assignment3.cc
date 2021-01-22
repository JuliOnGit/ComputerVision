///
///  Assignment 3
///  Face Verification
///
#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <fstream>

#include "face.h"
#include "ROC.h"
using namespace std;

vector<pair<string,string>> getPairs(string path) {
	
	
	vector<string> people;

	namespace fs = boost::filesystem; 
	for (fs::directory_iterator it(fs::path(path.c_str())); it!=fs::directory_iterator(); it++) {
		if (is_regular_file(*it) and it->path().filename().stem().string().back()=='1') {
			string filename = it->path().filename().string();
			
			people.push_back(filename.substr(0,filename.size()-5));
		}
	}
	random_shuffle(people.begin(), people.end());
	
	vector<pair<string,string>> pairs;
	for (uint i = 0; i < people.size()/2; i++) 
		pairs.push_back({people[i], people[(i+1)%(people.size()/2)]});
	for (uint i = people.size()/2; i < people.size(); i++) 
		pairs.push_back({people[i], people[i]});
		
	random_shuffle(pairs.begin(), pairs.end());
		
	return pairs;
}

int main(int argc, char* argv[]) {
	
	
	//// parse command line options
	boost::program_options::variables_map pom;
	{
		namespace po = boost::program_options;
		po::options_description pod(string("Allowed options for ")+argv[0]);
		pod.add_options() 
			("help,h", "produce this help message")
			("gui,g", "Enable the GUI");

		po::store(po::command_line_parser( argc, argv ).options(pod).run(), pom);
		po::notify(pom);

		if (pom.count("help")) {
			cout << "Usage:" << endl <<  pod << "\n";
			return 0;
		}
	}
	

	//// TRAIN
	
	/// create person classification model instance
	FACE model;

	/// train model with all images in the train folder
	cout << "Start Training" << endl;
	model.startTraining();
		
	for (auto p : getPairs("data/train")) {
		cout << "Train on Person: " << p.first << " and " << p.second << endl;
		model.train( cv::imread("data/train/"+ p.first+"1.jpg",-1), cv::imread("data/train/"+p.second+"2.jpg",-1), p.first == p.second );
	}
	
	cout << "Finish Training" << endl;
	model.finishTraining();
	
	//// VALIDATION

	/// validate model with all images in the validation folder, 
	ROC<double> roc;
	for (auto p :  getPairs("data/validation")) {
		double hyp = model.verify( cv::imread("data/validation/"+p.first+"1.jpg",-1), cv::imread("data/validation/"+p.second+"2.jpg",-1));
		roc.add(p.first == p.second, hyp);
		cout << "Validation: " << p.first << " and " << p.second << ": " << hyp << endl;
	}
	
	/// After testing, update statistics and show results
	roc.update();
	
	cout << "Best EER score: " << roc.EER << endl;
	
	/// Display final result if desired
	if (pom.count("gui")) {
		cv::imshow("ROC", roc.draw());
		cv::waitKey(0);
	}	

	//// TEST

	/// apply model to all images pairs in the test folder, 
	ofstream oss("result.txt");
	{
		ifstream iff("data/testPairs");
		string s1, s2;
		while (iff >> s1 >> s2) {
			double hyp = model.verify( cv::imread("data/test/"+s1,-1), cv::imread("data/test/"+s2,-1));
			oss << hyp << endl;
			cout << "Test: " << s1 << " and " << s2 << ": " << hyp << endl;
		}
	}	
}

