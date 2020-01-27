CXXFLAGS += -O3 -std=c++11 -Wall -Wpedantic -fPIC

ifdef SYSTEMROOT
    CXXFLAGS += -lpthread
else
    CXXFLAGS += -pthread
endif

ifeq ($(SSE4),1)
	CXXFLAGS += -msse4.2
endif

SRCS := $(wildcard omp/*.cpp)
OBJS := ${SRCS:.cpp=.o}

all: lib/ompeval.o test

lib:
	mkdir lib

lib/ompeval.o: $(OBJS) | lib
	ar rcs $@ $^

test: test.cpp benchmark.cpp lib/ompeval.o
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	$(RM) test test.exe lib/ompeval.o $(OBJS)
