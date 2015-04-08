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

#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <fstream>

#include <utils.hpp>
#include <Observation.hpp>


#ifndef SNR_HPP
#define SNR_HPP

namespace PulsarSearch {

class snrDedispersedConf {
public:
  snrDedispersedConf();
  ~snrDedispersedConf();
  // Get
  unsigned int getNrSamplesPerBlock() const;
  unsigned int getNrSamplesPerThread() const;
  // Set
  void setNrSamplesPerBlock(unsigned int samples);
  void setNrSamplesPerThread(unsigned int samples);
  // utils
  std::string print() const;

private:
  unsigned int nrSamplesPerBlock;
  unsigned int nrSamplesPerThread;
};

typedef std::map< std::string, std::map< unsigned int, PulsarSearch::snrDedispersedConf > > tunedSNRDedispersedConf;

// OpenCL SNR
std::string * getSNRDedispersedOpenCL(const snrDedispersedConf & conf, const std::string & dataType, const AstroData::Observation & observation);
// Read configuration files
void readTunedSNRDedispersedConf(tunedSNRDedispersedConf & tunedSNR, const std::string & snrFilename);


// Implementations
inline unsigned int snrDedispersedConf::getNrSamplesPerBlock() const {
  return nrSamplesPerBlock;
}

inline unsigned int snrDedispersedConf::getNrSamplesPerThread() const {
  return nrSamplesPerThread;
}

inline void snrDedispersedConf::setNrSamplesPerBlock(unsigned int samples) {
  nrSamplesPerBlock = samples;
}

inline void snrDedispersedConf::setNrSamplesPerThread(unsigned int samples) {
  nrSamplesPerThread = samples;
}

} // PulsarSearch

#endif // SNR_HPP

