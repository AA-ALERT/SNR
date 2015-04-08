// Copyright 2014 Alessio Sclocco <a.sclocco@vu.nl>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <string>
#include <vector>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <SNR.hpp>
#include <utils.hpp>
#include <Timer.hpp>
#include <Stats.hpp>

typedef float dataType;
std::string typeName("float");


void initializeDeviceMemoryD(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< dataType > * dedispersedData, cl::Buffer * dedispersedData_d, cl::Buffer * snrData_d, const unsigned int snrData_size);

int main(int argc, char * argv[]) {
  bool reInit = true;
	unsigned int nrIterations = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
  unsigned int threadUnit = 0;
  unsigned int threadInc = 0;
	unsigned int minThreads = 0;
	unsigned int maxItemsPerThread = 0;
	unsigned int maxColumns = 0;
  AstroData::Observation observation;
  PulsarSearch::snrDedispersedConf dConf;
  cl::Event event;

	try {
    isa::utils::ArgumentList args(argc, argv);

		nrIterations = args.getSwitchArgument< unsigned int >("-iterations");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
		observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    threadUnit = args.getSwitchArgument< unsigned int >("-thread_unit");
    threadInc = args.getSwitchArgument< unsigned int >("-thread_inc");
		minThreads = args.getSwitchArgument< unsigned int >("-min_threads");
		maxItemsPerThread = args.getSwitchArgument< unsigned int >("-max_items");
		maxColumns = args.getSwitchArgument< unsigned int >("-max_columns");
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
	} catch ( isa::utils::EmptyCommandLine & err ) {
		std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... -padding ... -thread_unit ... -thread_inc ... -min_threads ... -max_threads ... -max_items ... -max_columns ... -dms ... -samples ..." << std::endl;
		return 1;
	} catch ( std::exception & err ) {
		std::cerr << err.what() << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context clContext;
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = 0;

	// Allocate memory
  std::vector< dataType > dedispersedData;
  cl::Buffer dedispersedData_d, snrData_d;

  dedispersedData.resize(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());

	srand(time(0));
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      dedispersedData[(dm * observation.getNrSamplesPerPaddedSecond()) + sample] = static_cast< dataType >(rand() % 10);
    }
  }

	// Find the parameters
	std::vector< unsigned int > samplesPerBlock;
	for ( unsigned int samples = minThreads; samples <= maxColumns; samples += threadInc ) {
		if ( (observation.getNrSamplesPerPaddedSecond() % samples) == 0 || (observation.getNrSamplesPerSecond() % samples) == 0 ) {
			samplesPerBlock.push_back(samples);
		}
	}

	std::cout << std::fixed << std::endl;
  std::cout << "# nrDMs nrSamples samplesPerBlock samplesPerThread GB/s time stdDeviation COV" << std::endl << std::endl;

  for ( std::vector< unsigned int >::iterator samples = samplesPerBlock.begin(); samples != samplesPerBlock.end(); ++samples ) {
    if ( *samples % threadUnit != 0 ) {
      continue;
    }
    dConf.setNrSamplesPerBlock(*samples);

    for ( unsigned int samplesPerThread = 1; 5 + (4 * samplesPerThread) < maxItemsPerThread; samplesPerThread++ ) {
      if ( (observation.getNrSamplesPerPaddedSecond() % (dConf.getNrSamplesPerBlock() * samplesPerThread)) == 0 || (observation.getNrSamplesPerSecond() % (dConf.getNrSamplesPerBlock() * samplesPerThread)) == 0 ) {
        continue;
      }
      dConf.setNrSamplesPerThread(samplesPerThread);

      // Generate kernel
      double gbs = isa::utils::giga((static_cast< long long unsigned int >(observation.getNrDMs()) * observation.getNrSamplesPerSecond() * sizeof(dataType)) + (static_cast< long long unsigned int >(observation.getNrDMs()) * sizeof(dataType)));
      cl::Kernel * kernel;
      isa::utils::Timer timer;
      std::string * code = PulsarSearch::getSNRDedispersedOpenCL(dConf, typeName, observation);

      if ( reInit ) {
        delete clQueues;
        clQueues = new std::vector< std::vector< cl::CommandQueue > >();
        isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, &clContext, clDevices, clQueues);
        try {
          initializeDeviceMemoryD(clContext, &(clQueues->at(clDeviceID)[0]), &dedispersedData, &dedispersedData_d, &snrData_d, observation.getNrDMs() * sizeof(float));
        } catch ( cl::Error & err ) {
          return -1;
        }
        reInit = false;
      }
      try {
        kernel = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", clContext, clDevices->at(clDeviceID));
      } catch ( isa::OpenCL::OpenCLError & err ) {
        std::cerr << err.what() << std::endl;
        delete code;
        break;
      }
      delete code;

      cl::NDRange global;
      if ( observation.getNrSamplesPerSecond() % (dConf.getNrSamplesPerBlock() * dConf.getNrSamplesPerThread()) == 0 ) {
      global = cl::NDRange(observation.getNrSamplesPerSecond() / dConf.getNrSamplesPerThread(), observation.getNrDMs());
      } else {
      global = cl::NDRange(observation.getNrSamplesPerPaddedSecond() / dConf.getNrSamplesPerThread(), observation.getNrDMs());
      }
      cl::NDRange local = cl::NDRange(dConf.getNrSamplesPerBlock(), 1);

      kernel->setArg(0, dedispersedData_d);
      kernel->setArg(1, snrData_d);

      try {
        // Warm-up run
        clQueues->at(clDeviceID)[0].finish();
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
        event.wait();
        // Tuning runs
        for ( unsigned int iteration = 0; iteration < nrIterations; iteration++ ) {
          timer.start();
          clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
          event.wait();
          timer.stop();
        }
      } catch ( cl::Error & err ) {
        std::cerr << "OpenCL error kernel execution (";
        std::cerr << dConf.print();
        std::cerr << "): " << isa::utils::toString(err.err()) << "." << std::endl;
        delete kernel;
        if ( err.err() == -4 || err.err() == -61 ) {
          return -1;
        }
        reInit = true;
        break;
      }
      delete kernel;

      std::cout << observation.getNrDMs() << " " << observation.getNrSamplesPerSecond() << " ";
      std::cout << dConf.print() << " ";
      std::cout << std::setprecision(3);
      std::cout << gbs / timer.getAverageTime() << " ";
      std::cout << std::setprecision(6);
      std::cout << timer.getAverageTime() << " " << timer.getStandardDeviation() << " " << timer.getCoefficientOfVariation() << std::endl;
    }
  }

	std::cout << std::endl;

	return 0;
}

void initializeDeviceMemoryD(cl::Context & clContext, cl::CommandQueue * clQueue, std::vector< dataType > * dedispersedData, cl::Buffer * dedispersedData_d, cl::Buffer * snrData_d, const unsigned int snrData_size) {
  try {
    *dedispersedData_d = cl::Buffer(clContext, CL_MEM_READ_WRITE, dedispersedData->size() * sizeof(dataType), 0, 0);
    *snrData_d = cl::Buffer(clContext, CL_MEM_WRITE_ONLY, snrData_size, 0, 0);
    clQueue->enqueueWriteBuffer(*dedispersedData_d, CL_FALSE, 0, dedispersedData->size() * sizeof(dataType), reinterpret_cast< void * >(dedispersedData->data()));
    clQueue->finish();
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    throw;
  }
}

