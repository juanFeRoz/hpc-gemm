CXX 			= g++
CXXFLAGS 	= -O3 -march=native -fopenmp -ffast-math -Wall -Wextra -Werror
LDFLAGS 	= -fopenmp

TARGET 		= gemm

SRCS 			= main.cpp 
OBJS 			= $(SRCS:.cpp=.o)
HDRS 			= Matrix.hpp Kernels.hpp

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(LDFLAGS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
