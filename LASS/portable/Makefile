# Standalone build for the portable synthesis layer + its parity test.
# Links against the already-built lib/liblass.a (make lass config=release).
# Independent of premake/CMOD (see plan flaw F8).

CXX      ?= g++
CXXFLAGS ?= -std=c++11 -O2 -Wall -I../src
LASSLIB   = ../../lib/liblass.a
CUDA_LIB  = /usr/local/cuda/lib64
LDLIBS    = $(LASSLIB) -L$(CUDA_LIB) -lcudart -lpthread -lm

# libsndfile is only needed if AuWriter is pulled in; add if the linker asks.
SNDFILE  ?= -lsndfile

test_partial_parity: test_partial_parity.cpp PortableSynth.cpp PortableSynth.h
	$(CXX) $(CXXFLAGS) test_partial_parity.cpp PortableSynth.cpp $(LDLIBS) $(SNDFILE) -o $@

# Microbenchmark: links the CUDA map (renderMapCuda) from liblass.a.
bench_map: bench_map.cpp PortableSynth.h
	$(CXX) $(CXXFLAGS) -DHAVE_CUDA bench_map.cpp $(LDLIBS) $(SNDFILE) -o $@

.PHONY: run bench clean
run: test_partial_parity
	./test_partial_parity

bench: bench_map
	./bench_map

clean:
	rm -f test_partial_parity bench_map
