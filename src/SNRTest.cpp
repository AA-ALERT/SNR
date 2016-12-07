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

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <Observation.hpp>
#include <InitializeOpenCL.hpp>
#include <Kernel.hpp>
#include <utils.hpp>
#include <SNR.hpp>
#include <Stats.hpp>


int main(int argc, char *argv[]) {
  bool printCode = false;
  bool printResults = false;
  unsigned int padding = 0;
	unsigned int clPlatformID = 0;
	unsigned int clDeviceID = 0;
	uint64_t wrongSamples = 0;
	AstroData::Observation observation;
  PulsarSearch::snrConf conf;

	try {
    isa::utils::ArgumentList args(argc, argv);
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_results");
		clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
		clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threads0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-items0"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
	} catch  ( isa::utils::SwitchNotFound &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-print_code] [-print_results] -opencl_platform ... -opencl_device ... -padding ... -threads0 ... -items0 ... -dms ... -samples ..." << std::endl;
		return 1;
	}

	// Initialize OpenCL
	cl::Context * clContext = new cl::Context();
	std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
	std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
	std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

	// Allocate memory
  std::vector< inputDataType > input, output;
  cl::Buffer input_d, output_d;
  input.resize(observation.getNrDMs() * observation.getNrSamplesPerPaddedSecond(padding / sizeof(inputDataType)));
  output.resize(observation.getNrPaddedDMs(padding / sizeof(inputDataType)));
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(inputDataType), 0, 0);
    output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(float), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
      input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(inputDataType))) + sample] = static_cast< inputDataType >(rand() % 10);
      if ( printResults ) {
        std::cout << input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(inputDataType))) + sample] << " ";
      }
    }
    if ( printResults ) {
      std::cout << std::endl;
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(inputDataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code;
  code = PulsarSearch::getSNRDMsSamplesOpenCL< inputDataType >(conf, inputDataName, observation.getNrSamplesPerSecond(), padding);
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    kernel = isa::OpenCL::compile("snrDMsSamples" + isa::utils::toString(observation.getNrSamplesPerSecond()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  std::vector< isa::utils::Stats< inputDataType > > control(observation.getNrDMs());
  try {
    cl::NDRange global;
    cl::NDRange local;

    global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrDMs());
    local = cl::NDRange(conf.getNrThreadsD0(), 1);

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);

    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(float), reinterpret_cast< void * >(output.data()));
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      control[dm] = isa::utils::Stats< inputDataType >();

      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
        control[dm].addElement(input[(dm * observation.getNrSamplesPerPaddedSecond(padding / sizeof(inputDataType))) + sample]);
      }
    }
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    if ( ! isa::utils::same(output[dm], (control[dm].getMax() - control[dm].getMean()) / control[dm].getStandardDeviation()) ) {
      wrongSamples++;
      if ( printResults ) {
        std::cout << "**" << output[dm] << " != " << (control[dm].getMax() - control[dm].getMean()) / control[dm].getStandardDeviation() << "** ";
      }
    } else if (printResults ) {
      std::cout << output[dm] << " ";
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast< uint64_t >(observation.getNrDMs()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	// Allocate memory
  wrongSamples = 0;
  input.resize(observation.getNrSamplesPerSecond() * observation.getNrPaddedDMs(padding / sizeof(inputDataType)));
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(inputDataType), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

	srand(time(0));
  for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      input[(sample * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + dm] = static_cast< inputDataType >(rand() % 10);
      if ( printResults ) {
        std::cout << input[(sample * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + dm] << " ";
      }
    }
    if ( printResults ) {
      std::cout << std::endl;
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(inputDataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  code = PulsarSearch::getSNRSamplesDMsOpenCL< inputDataType >(conf, inputDataName, observation, padding);
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    kernel = isa::OpenCL::compile("snrSamplesDMs" + isa::utils::toString(observation.getNrDMs()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
  } catch ( isa::OpenCL::OpenCLError &err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  try {
    cl::NDRange global;
    cl::NDRange local;

    global = cl::NDRange(observation.getNrDMs() / conf.getNrItemsD0());
    local = cl::NDRange(conf.getNrThreadsD0());

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);

    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(float), reinterpret_cast< void * >(output.data()));
    for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
      control[dm] = isa::utils::Stats< inputDataType >();

      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerSecond(); sample++ ) {
        control[dm].addElement(input[(sample * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + dm]);
      }
    }
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error: " << isa::utils::toString< cl_int >(err.err()) << "." << std::endl;
    return 1;
  }

  for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
    if ( ! isa::utils::same(output[dm], (control[dm].getMax() - control[dm].getMean()) / control[dm].getStandardDeviation()) ) {
      wrongSamples++;
      if ( printResults ) {
        std::cout << "**" << output[dm] << " != " << (control[dm].getMax() - control[dm].getMean()) / control[dm].getStandardDeviation() << "** ";
      }
    } else if (printResults ) {
      std::cout << output[dm] << " ";
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast< uint64_t >(observation.getNrDMs()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

	return 0;
}

