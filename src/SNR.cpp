// Copyright 2015 Alessio Sclocco <a.sclocco@vu.nl>
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

#include <SNR.hpp>

namespace PulsarSearch {

snrDMsSamplesConf::snrDMsSamplesConf() {}

snrDMsSamplesConf::~snrDMsSamplesConf() {}

std::string snrDMsSamplesConf::print() const {
  return isa::utils::toString(nrSamplesPerBlock) + " " + isa::utils::toString(nrSamplesPerThread);
}

void readTunedSNRDMsSamplesConf(tunedSNRDMsSamplesConf & tunedSNR, const std::string & snrFilename) {
  std::string temp;
  std::ifstream snrFile(snrFilename);
  while ( ! snrFile.eof() ) {
    unsigned int splitPoint = 0;
    std::getline(snrFile, temp);
    if ( ! std::isalpha(temp[0]) ) {
      continue;
    }
    std::string deviceName;
    unsigned int nrDMs = 0;
    unsigned int nrSamples = 0;
    PulsarSearch::snrDMsSamplesConf parameters;

    splitPoint = temp.find(" ");
    deviceName = temp.substr(0, splitPoint);
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    nrDMs = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    nrSamples = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters.setNrSamplesPerBlock(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    parameters.setNrSamplesPerThread(isa::utils::castToType< std::string, unsigned int >(temp));

		if ( tunedSNR.count(deviceName) == 0 ) {
      std::map< unsigned int, std::map< unsigned int, PulsarSearch::snrDMsSamplesConf > > externalContainer;
      std::map< unsigned int, PulsarSearch::snrDMsSamplesConf > internalContainer;

			internalContainer.insert(std::make_pair(nrSamples, parameters));
			externalContainer.insert(std::make_pair(nrDMs, internalContainer));
			tunedSNR.insert(std::make_pair(deviceName, externalContainer));
		} else if ( tunedSNR[deviceName].count(nrDMs) == 0 ) {
      std::map< unsigned int, PulsarSearch::snrDMsSamplesConf > internalContainer;

			internalContainer.insert(std::make_pair(nrSamples, parameters));
			tunedSNR[deviceName].insert(std::make_pair(nrDMs, internalContainer));
		} else {
			tunedSNR[deviceName][nrSamples].insert(std::make_pair(nrSamples, parameters));
		}
  }
}

} // PulsarSearch

