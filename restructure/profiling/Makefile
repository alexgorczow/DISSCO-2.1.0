# Build the standalone stage micro-profiler.
#
#   make            -> ./profile_stages   (needs liblass.a already built:
#                       from repo root:  make lass config=release)
#   make clean
#
# The sweep driver (benchmark.py) and the in-pipeline StageProfiler.h need no
# build of their own -- the profiler is compiled into cmod. This Makefile is
# only for the isolated synthetic-Sound micro-profiler.

REPO := ../..
LIB  := $(REPO)/lib
CXX  := g++
CXXFLAGS := -O3 -std=c++14 -I$(REPO)/LASS/src
LDLIBS := -L$(LIB) -L/usr/local/cuda/lib64 \
          -Wl,--start-group -llass -Wl,--end-group \
          -lpthread -lsndfile -lcudart

profile_stages: profile_stages.cpp $(LIB)/liblass.a
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f profile_stages

.PHONY: clean
