
SOURCE_ROOT ?= $(HOME)

# https://github.com/isazi/utils
UTILS := $(SOURCE_ROOT)/src/utils
# https://github.com/isazi/OpenCL
OPENCL := $(SOURCE_ROOT)/src/OpenCL
# https://github.com/isazi/AstroData
ASTRODATA := $(SOURCE_ROOT)/src/AstroData

INCLUDES := -I"include" -I"$(ASTRODATA)/include" -I"$(UTILS)/include"
CL_INCLUDES := $(INCLUDES) -I"$(OPENCL)/include"
CL_LIBS := -L"$(OPENCL_LIB)"

CFLAGS := -std=c++11 -Wall
ifneq ($(debug), 1)
	CFLAGS += -O3 -g0
else
	CFLAGS += -O0 -g3
endif

LDFLAGS := -lm
CL_LDFLAGS := $(LDFLAGS) -lOpenCL

CC := g++

# Dependencies
DEPS := $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/ArgumentList.o $(UTILS)/bin/Timer.o $(UTILS)/bin/utils.o bin/SNR.o
CL_DEPS := $(DEPS) $(OPENCL)/bin/Exceptions.o $(OPENCL)/bin/InitializeOpenCL.o $(OPENCL)/bin/Kernel.o 


all: bin/SNR.o bin/SNRTest bin/SNRTuning bin/printCode

bin/SNR.o: $(ASTRODATA)/bin/Observation.o $(UTILS)/bin/utils.o include/SNR.hpp src/SNR.cpp
	-@mkdir -p bin
	$(CC) -o bin/SNR.o -c src/SNR.cpp $(CL_INCLUDES) $(CFLAGS)

bin/SNRTest: $(CL_DEPS) src/SNRTest.cpp
	-@mkdir -p bin
	$(CC) -o bin/SNRTest src/SNRTest.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/SNRTuning: $(CL_DEPS) src/SNRTuning.cpp
	-@mkdir -p bin
	$(CC) -o bin/SNRTuning src/SNRTuning.cpp $(CL_DEPS) $(CL_INCLUDES) $(CL_LIBS) $(CL_LDFLAGS) $(CFLAGS)

bin/printCode: $(DEPS) src/printCode.cpp
	-@mkdir -p bin
	$(CC) -o bin/printCode src/printCode.cpp $(DEPS) $(INCLUDES) $(LDFLAGS) $(CFLAGS)

test: bin/printCode bin/SNRTest
	./bin/printCode -dms_samples -padding 32 -threads0 16 -items0 16 -samples 256 -dms 256
	./bin/SNRTest -opencl_platform 0 -opencl_device 0 -padding 32 -threads0 16 -items0 16 -dms 256 -samples 256

tune: bin/SNRTuning
	./bin/SNRTuning -opencl_platform 0 -opencl_device 0 -padding 32 -iterations 4 -max_threads 32 -max_items 32 -dms 256 -samples 256 -min_threads 1

clean:
	-@rm bin/*

