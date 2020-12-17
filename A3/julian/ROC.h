#include <opencv2/opencv.hpp>

template<typename T>
class ROC {
	
	static const size_t N = 512;
	
	std::map<T,std::pair<uint64,uint64>> PN;
public:
	std::vector<std::pair<long double, long double>> graph;
	long double F1, MCC, EER; 

	void add(bool value, T prediction) {
		if (value)
			PN[prediction].first++;
		else
			PN[prediction].second++;
	}

	void update() {
		
		long double TP=0, TN=0, FP=0 ,FN=0;
		for (auto &e : PN) {
			TP += e.second.first;
			FP += e.second.second;
		}

		F1=0;
		MCC = -1;
		EER = 1e10;
		graph.clear();
		graph.push_back({FP/double(FP+TN),TP/double(TP+FN)});
		F1 = std::max(F1, 2*TP/(2*TP+FP+FN));
		MCC = std::max(MCC, (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1e-10));
		
		for (auto &e : PN) {
			
			for (size_t i=0; i<N; i++) {
				TP -= e.second.first/double(N);
				FN += e.second.first/double(N);
				FP -= e.second.second/double(N);
				TN += e.second.second/double(N);
				
				long double FPR = FP/double(FP+TN);
				long double TPR = TP/double(TP+FN);
				graph.push_back({FPR,TPR});
				
				F1 = std::max(F1, 2*TP/(2*TP+FP+FN));
				MCC = std::max(MCC, (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+1e-10));
				EER = std::min(EER, std::max(1.-TPR,FPR));
			}
		}
	}
	
	cv::Mat3b draw(int sz=512) {
		
		cv::Mat3b roc(sz,sz,cv::Vec3b(255,255,255));
		
		for (auto &p : graph)
			cv::circle(roc,cv::Point2f(sz*p.first,sz*(1-p.second)),1.5, cv::Scalar(0,0,255), 1.5, cv::LINE_AA);

		return roc;
	}
};
