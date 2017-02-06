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
  bool DMsSamples = false;
  unsigned int padding = 0;
  unsigned int clPlatformID = 0;
  unsigned int clDeviceID = 0;
  uint64_t wrongSamples = 0;
  AstroData::Observation observation;
  PulsarSearch::snrConf conf;

  try {
    isa::utils::ArgumentList args(argc, argv);
    DMsSamples = args.getSwitch("-dms_samples");
    bool samplesDMs = args.getSwitch("-samples_dms");
    if ( (DMsSamples && samplesDMs) || (!DMsSamples && !samplesDMs) ) {
      std::cerr << "-dms_samples and -samples_dms are mutually exclusive." << std::endl;
      return 1;
    }
    printCode = args.getSwitch("-print_code");
    printResults = args.getSwitch("-print_results");
    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
    padding = args.getSwitchArgument< unsigned int >("-padding");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threadsD0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-itemsD0"));
    conf.setSubbandDedispersion(args.getSwitch("-subband"));
    observation.setNrSynthesizedBeams(args.getSwitchArgument< unsigned int >("-beams"));
    observation.setNrSamplesPerBatch(args.getSwitchArgument< unsigned int >("-samples"));
    if ( conf.getSubbandDedispersion() ) {
      observation.setDMSubbandingRange(args.getSwitchArgument< unsigned int >("-subbanding_dms"), 0.0f, 0.0f);
    } else {
      observation.setDMSubbandingRange(1, 0.0f, 0.0f);
    }
    observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
  } catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception &err ) {
    std::cerr << "Usage: " << argv[0] << " [-dms_samples | -samples_dms] [-print_code] [-print_results] -opencl_platform ... -opencl_device ... -padding ... -threadsD0 ... -itemsD0 ... [-subband] -beams ... -dms ... -samples ..." << std::endl;
    std::cerr << "\t -subband : -subbanding_dms ..." << std::endl;
    return 1;
  }

  // Initialize OpenCL
  cl::Context * clContext = new cl::Context();
  std::vector< cl::Platform > * clPlatforms = new std::vector< cl::Platform >();
  std::vector< cl::Device > * clDevices = new std::vector< cl::Device >();
  std::vector< std::vector< cl::CommandQueue > > * clQueues = new std::vector< std::vector < cl::CommandQueue > >();

  isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

  // Allocate memory
  std::vector< inputDataType > input;
  std::vector< float > output;
  cl::Buffer input_d, output_d;
  if ( DMsSamples ) {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMsSubbanding() * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType)));
  } else {
    input.resize(observation.getNrSynthesizedBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType)));
  }
  output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float)));
  try {
    input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(inputDataType), 0, 0);
    output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(float), 0, 0);
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error allocating memory: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  srand(time(0));
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    if ( printResults ) {
      std::cout << "Beam: " << beam << std::endl;
    }
    if ( DMsSamples ) {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMsSubbanding(); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          if ( printResults ) {
            std::cout << "DM: " << (subbandDM * observation.getNrDMs()) + dm << " -- ";
          }
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
            input[(beam * observation.getNrDMsSubbanding() * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + sample] = static_cast< inputDataType >(rand() % 10);
            if ( printResults ) {
              std::cout << input[(beam * observation.getNrDMsSubbanding() * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + sample] << " ";
            }
          }
          if ( printResults ) {
            std::cout << std::endl;
          }
        }
      }
    } else {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
        if ( printResults ) {
          std::cout << "Sample: " << sample << " -- ";
        }
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMsSubbanding(); subbandDM++ ) {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + (sample * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + dm] = static_cast< inputDataType >(rand() % 10);
            if ( printResults ) {
              std::cout << input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + (sample * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + dm] << " ";
            }
          }
          if ( printResults ) {
            std::cout << std::endl;
          }
        }
      }
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  // Copy data structures to device
  try {
    clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(inputDataType), reinterpret_cast< void * >(input.data()));
  } catch ( cl::Error &err ) {
    std::cerr << "OpenCL error H2D transfer: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }

  // Generate kernel
  cl::Kernel * kernel;
  std::string * code;
  if ( DMsSamples ) {
    code = PulsarSearch::getSNRDMsSamplesOpenCL< inputDataType >(conf, inputDataName, observation, observation.getNrSamplesPerBatch(), padding);
  } else {
    code = PulsarSearch::getSNRSamplesDMsOpenCL< inputDataType >(conf, inputDataName, observation, observation.getNrSamplesPerBatch(), padding);
  }
  if ( printCode ) {
    std::cout << *code << std::endl;
  }

  try {
    if ( DMsSamples ) {
      kernel = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    } else {
      kernel = isa::OpenCL::compile("snrSamplesDMs" + std::to_string(observation.getNrDMsSubbanding() * observation.getNrDMs()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
    }
  } catch ( isa::OpenCL::OpenCLError & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  }

  // Run OpenCL kernel and CPU control
  std::vector< isa::utils::Stats< inputDataType > > control(observation.getNrSynthesizedBeams() * observation.getNrDMsSubbanding() * observation.getNrDMs());
  try {
    cl::NDRange global;
    cl::NDRange local;

    if ( DMsSamples ) {
      global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrDMsSubbanding() * observation.getNrDMs(), observation.getNrSynthesizedBeams());
      local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
    } else {
      global = cl::NDRange((observation.getNrDMsSubbanding() * observation.getNrDMs()) / conf.getNrItemsD0(), observation.getNrSynthesizedBeams());
      local = cl::NDRange(conf.getNrThreadsD0(), 1);
    }

    kernel->setArg(0, input_d);
    kernel->setArg(1, output_d);

    clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
    clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(float), reinterpret_cast< void * >(output.data()));
  } catch ( cl::Error & err ) {
    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
    return 1;
  }
  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMsSubbanding(); subbandDM++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        control[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm] = isa::utils::Stats< inputDataType >();
      }
    }
    if ( DMsSamples ) {
      for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMsSubbanding(); subbandDM++ ) {
        for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
          for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
            control[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm].addElement(input[(beam * observation.getNrDMsSubbanding() * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + (subbandDM * observation.getNrDMs() * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerPaddedBatch(padding / sizeof(inputDataType))) + sample]);
          }
        }
      }
    } else {
      for ( unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++ ) {
        for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMsSubbanding(); subbandDM++ ) {
          for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
            control[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm].addElement(input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + (sample * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(inputDataType))) + dm]);
          }
        }
      }
    }
  }

  for ( unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++ ) {
    if ( printResults ) {
      std::cout << "Beam: " << beam << std::endl;
    }
    for ( unsigned int subbandDM = 0; subbandDM < observation.getNrDMsSubbanding(); subbandDM++ ) {
      for ( unsigned int dm = 0; dm < observation.getNrDMs(); dm++ ) {
        if ( !isa::utils::same(output[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm], (control[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm].getMax() - control[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm].getMean()) / control[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm].getStandardDeviation()) ) {
          wrongSamples++;
        }
        if ( printResults ) {
          std::cout << output[(beam * observation.getNrDMsSubbanding() * observation.getNrPaddedDMs(padding / sizeof(float))) + (subbandDM * observation.getNrPaddedDMs(padding / sizeof(float))) + dm] << " ";
        }
      }
      if ( printResults ) {
        std::cout << std::endl;
      }
    }
  }
  if ( printResults ) {
    std::cout << std::endl;
  }

  if ( wrongSamples > 0 ) {
    std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast< uint64_t >(observation.getNrSynthesizedBeams() * observation.getNrDMsSubbanding() * observation.getNrDMs()) << "%)." << std::endl;
  } else {
    std::cout << "TEST PASSED." << std::endl;
  }

  return 0;
}

