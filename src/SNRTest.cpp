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
#include <ctime>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <SNR.hpp>
#include <Stats.hpp>

typedef float dataType;
std::string typeName("float");


int main(int argc, char *argv[]) {
  bool printCode = false;
  bool printResults = false;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	long long unsigned int wrongSamples = 0;
	AstroData::Observation observation;
  PulsarSearch::snrDedispersedConf dConf;

	try {
    isa::utils::ArgumentList args(argc, argv);
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_res");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    dConf.setNrSamplesPerBlock(args.getSwitchArgument< unsigned int >("-sb"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print_code] [-print_res] -opencl_platform ... -opencl_device ... -padding ... -sb ... -dms ... -samples ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
  std::vector< dataType > dedispersedData, snrData;
  cl::Buffer dedispersedData_d, snrData_d;
  dedispersedData.resize(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond());
  snrData.resize(observation.getNrPaddedDMs());
  try {
    dedispersedData_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, dedispersedData.size() * sizeof(dataType), 0, 0);
    snrData_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, snrData.size() * sizeof(float), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      dedispersedData[(dm * observation.getNrSamplesPerPaddedSecond()) + sample] = static_cast< dataType >(rand() % 10);
    }
  }
  std::fill(snrData.begin(), snrData.end(), 0.0f);

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(dedispersedData_d, CL_FALSE, 0, dedispersedData.size() * sizeof(dataType), reinterpret_cast< void * >(dedispersedData.data()));
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(snrData_d, CL_FALSE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code;
  code = PulsarSearch::getSNRDedispersedOpenCL(dConf, typeName, observation);
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    kernel = isa::OpenCL::compile("snrDedispersed", *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  std::vector< isa::utils::Stats< dataType > > control(observation.getNrDMs());
  try {
    cl::NDRange global;
    cl::NDRange local;

    global = cl::NDRange(dConf.getNrSamplesPerBlock(), observation.getNrDMs());
    local = cl::NDRange(dConf.getNrSamplesPerBlock(), 1);

    kernel->setArg(0, dedispersedData_d);
    kernel->setArg(1, snrData_d);
    
    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(snrData_d, CL_TRUE, 0, snrData.size() * sizeof(float), reinterpret_cast< void * >(snrData.data()));
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      control[dm] = isa::utils::Stats< dataType >();

      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
        control[dm].addElement(dedispersedData[(dm * observation.getNrSamplesPerPaddedSecond()) + sample]);
      }
    }
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    if ( ! isa::utils::same(snrData[dm], (control[dm].getMax() - control[dm].getMean()) / control[dm].getStandardDeviation()) ) {
      wrongSamples++;
      if ( printResults ) {
        std::cout << "**" << snrData[dm] << " != " << (control[dm].getMax() - control[dm].getMean()) / control[dm].getStandardDeviation() << "** ";
      }
    } else if (printResults ) {
      std::cout << snrData[dm] << " ";
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast< long long unsigned int >(observation.getNrDMs()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

