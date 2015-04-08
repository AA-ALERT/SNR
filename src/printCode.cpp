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
#include <utils.hpp>
#include <SNR.hpp>


int main(int argc, char *argv[]) {
  std::string typeName;
	AstroData::Observation observation;
  PulsarSearch::snrDedispersedConf dConf;

	try {
    isa::utils::ArgumentList args(argc, argv);
    typeName = args.getSwitchArgument< std::string >("-type");
    observation.setPadding(args.getSwitchArgument< unsigned int >("-padding"));
    dConf.setNrSamplesPerBlock(args.getSwitchArgument< unsigned int >("-sb"));
    dConf.setNrSamplesPerThread(args.getSwitchArgument< unsigned int >("-st"));
    observation.setNrSamplesPerSecond(args.getSwitchArgument< unsigned int >("-samples"));
		observation.setDMRange(args.getSwitchArgument< unsigned int >("-dms"), 0.0, 0.0);
	} catch  ( isa::utils::SwitchNotFound & err ) {
    std::cerr << err.what() << std::endl;
    return 1;
  } catch ( std::exception & err ) {
    std::cerr << "Usage: " << argv[0] << " -type ... -padding ... -sb ... -st ... -dms ... -samples ..." << std::endl;
		return 1;
	}

  // Generate kernel
  std::string * code;
  code = PulsarSearch::getSNRDedispersedOpenCL(dConf, typeName, observation);
  std::cout << *code << std::endl;

	return 0;
}

