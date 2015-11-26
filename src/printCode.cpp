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

#include <configuration.hpp>

#include <ArgumentList.hpp>
#include <utils.hpp>
#include <SNR.hpp>


int main(int argc, char *argv[]) {
  bool DMsSamples = false;
  bool samplesDMs = false;
  unsigned int padding = 0;
  AstroData::Observation observation;
  PulsarSearch::snrConf conf;

	try {
    isa::utils::ArgumentList args(argc, argv);
    DMsSamples = args.getSwitch("-dms_samples");
    samplesDMs = args.getSwitch("-samples_dms");
    if ( samplesDMs  ) {
      observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0f, 0.0f);
    }
    padding = args.getSwitchArgument< unsigned int >("-padding");
    conf.setNrThreadsD0(args.getSwitchArgument< unsigned int >("-threads0"));
    conf.setNrItemsD0(args.getSwitchArgument< unsigned int >("-items0"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
	} catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << " [-dms_samples] [-samples_dms] -padding ... -threads0 ... -items0 ... -samples ..." << std::endl;
    std::cerr << "\t -samples_dms -dms ..." << std::endl;
		return 1;
	}

  // Generate kernel
  std::string * code;
  if ( DMsSamples ) {
    code = PulsarSearch::getSNRDMsSamplesOpenCL< inputDataType >(conf, inputDataName, observation.getNrSamplesPerSecond(), padding);
    std::cout << *code << std::endl;
  }
  if ( samplesDMs ) {
    code = PulsarSearch::getSNRSamplesDMsOpenCL< inputDataType >(conf, inputDataName, observation, padding);
    std::cout << *code << std::endl;
  }

	return 0;
}

