METAL_CPP_DIR = metal-cpp_macOS14.2_iOS17.2

CXX = clang++
CXXFLAGS = -std=c++20 -O2 -I$(METAL_CPP_DIR) -I../include -framework Foundation -framework Metal -framework MetalKit

all : dot_product

dot_product : dot_product-build
	./dot_product-build

dot_product-build : dot_product.cpp Makefile
	$(CXX) -o $@ $(CXXFLAGS) dot_product.cpp

clean:
	rm -rf *-build

