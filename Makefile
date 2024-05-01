# Define the C++ compiler
CXX = g++

# Define any compile-time flags
CXXFLAGS = -Wall -std=c++11 $(shell pkg-config --cflags opencv4)

# Define any libraries to link into executable
LDFLAGS = $(shell pkg-config --libs opencv4)

# Define the source files, header files, and the resultant executable
SRCS = ZhangWan24.cpp
HEADERS = ZhangWan24.hpp
OBJS = $(SRCS:.cpp=.o)
MAIN = ZhangWan24

.PHONY: depend clean

all: $(MAIN)
	@echo  Compiled $(MAIN) successfully!

$(MAIN): $(OBJS) 
	$(CXX) $(CXXFLAGS) -o $(MAIN) $(OBJS) $(LDFLAGS)

# This is a suffix replacement rule for building .o's from .cpp's
# It uses automatic variables $<: the name of the prerequisite of the rule (a .cpp file)
# and $@: the name of the target of the rule (a .o file)
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
	makedepend $(CXXFLAGS) $^
