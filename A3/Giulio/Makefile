FLAGS += -std=c++0x
FLAGS += -std=c++11

FLAGS += -Wall -Wextra -pedantic
FLAGS += -Ofast
#FLAGS += -D__OPENCV_OBJDETECT_HPP__
#FLAGS +=  `pkg-config opencv --cflags --libs`
FLAGS +=  `pkg-config opencv4 --cflags --libs`
FLAGS +=  -lboost_system -lboost_program_options -lboost_serialization -lboost_filesystem

assignment3: *.cc *.h
	$(CXX) -o $@ assignment3.cc face.cc $(FLAGS)

.PHONY : clean

clean:
	rm -f assignment3 graph.txt ROC.png result.txt
