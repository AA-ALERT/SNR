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

snrDedispersedConf::snrDedispersedConf() {}

snrDedispersedConf::~snrDedispersedConf() {}

std::string snrDedispersedConf::print() const {
  return isa::utils::toString(nrSamplesPerBlock) + " " + isa::utils::toString(nrSamplesPerThread);
}
std::string * getSNRDedispersedOpenCL(const snrDedispersedConf & conf, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void snrDedispersed(__global const " + dataType + " * const restrict dedispersedData, __global float * const restrict snrData) {\n"
    "unsigned int dm = get_group_id(1);\n"
    "__local float reductionCOU[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "__local " + dataType + " reductionMAX[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "__local float reductionMEA[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "__local float reductionVAR[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "<%DEF%>"
    "\n"
    "// Compute phase\n"
    "for ( unsigned int sample = get_local_id(0) + " + isa::utils::toString(conf.getNrSamplesPerBlock() * conf.getNrSamplesPerThread()) + "; sample < " + isa::utils::toString(observation.getNrSamplesPerSecond()) + "; sample += " + isa::utils::toString(conf.getNrSamplesPerBlock() * conf.getNrSamplesPerThread()) + " ) {\n"
    + dataType + " item = 0;\n"
    "float delta = 0.0f;\n"
    "<%COMPUTE%>"
    "}\n"
    "// In-thread reduce\n"
    "// Local memory store\n"
    "reductionCOU[get_local_id(0)] = counter;\n"
    "reductionMAX[get_local_id(0)] = max;\n"
    "reductionMEA[get_local_id(0)] = mean;\n"
    "reductionVAR[get_local_id(0)] = variance;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Reduce phase\n"
    "unsigned int threshold = " + isa::utils::toString(conf.getNrSamplesPerBlock() / 2) + ";\n"
    "for ( sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( sample < threshold ) {\n"
    "float delta = reductionMEA[sample + threshold] - mean;\n"
    "counter += reductionCOU[sample + threshold];\n"
    "max = fmax(max, reductionMAX[sample + threshold]);\n"
    "mean = ((reductionCOU[sample] * mean) + (reductionCOU[sample + threshold] * reductionMEA[sample + threshold])) / counter;\n"
    "variance += reductionVAR[sample + threshold] + ((delta * delta) * ((reductionCOU[sample] * reductionCOU[sample + threshold]) / counter));\n"
    "reductionCOU[get_local_id(0)] = counter;\n"
    "reductionMAX[get_local_id(0)] = max;\n"
    "reductionMEA[get_local_id(0)] = mean;\n"
    "reductionVAR[get_local_id(0)] = variance;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "// Store\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "snrData[dm] = (max - mean) / native_sqrt(variance / (counter - 1));\n"
    "}\n"
    "}\n";
  std::string defTemplate = "float counter<%NUM%> = 1.0f;\n"
    + dataType + " max<%NUM%> = dedispersedData[(dm * " + isa::utils::toString(observation.getNrSamplesPerPaddedSecond()) + ") + (get_local_id(0) + <%OFFSET%>)];\n"
    "float variance<%NUM%> = 0.0f;\n"
    "float mean<%NUM%> = max;\n";
  std::string computeTemplate = "item = dedispersedData[(dm * " + isa::utils::toString(observation.getNrSamplesPerPaddedSecond()) + ") + (sample + <%OFFSET%>)];\n"
    "counter<%NUM%> += 1.0f;\n"
    "delta = item - mean<%NUM%>;\n"
    "max<%NUM%> = fmax(max<%NUM%>, item);\n"
    "mean<%NUM%> += delta / counter<%NUM%>;\n"
    "variance<%NUM%> += delta * (item - mean<%NUM%>);\n";
  // End kernel's template

  std::string * def_s = new std::string();
  std::string * compute_s = new std::string();

  for ( unsigned int sample = 0; sample < conf.getNrSamplesPerThread(); sample++ ) {
    std::string sample_s = isa::utils::toString(sample);
    std::string * temp = 0;

    temp = isa::utils::replace(&defTemplate, "<%NUM%>", sample_s);
    def_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&computeTemplate, "<%NUM%>", sample_s);
    compute_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  delete def_s;
  delete compute_s;

  return code;
}

void readTunedSNRDedispersedConf(tunedSNRDedispersedConf & tunedSNR, const std::string & snrFilename) {
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
    PulsarSearch::snrDedispersedConf parameters;
    splitPoint = temp.find(" ");
    deviceName = temp.substr(0, splitPoint);
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    nrDMs = isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters.setNrSamplesPerBlock(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    temp = temp.substr(splitPoint + 1);
    splitPoint = temp.find(" ");
    parameters.setNrSamplesPerThread(isa::utils::castToType< std::string, unsigned int >(temp.substr(0, splitPoint)));
    if ( tunedSNR.count(deviceName) == 0 ) {
      std::map< unsigned int, PulsarSearch::snrDedispersedConf > container;
      container.insert(std::make_pair(nrDMs, parameters));
      tunedSNR.insert(std::make_pair(deviceName, container));
    } else {
      tunedSNR[deviceName].insert(std::make_pair(nrDMs, parameters));
    }
  }
}

} // PulsarSearch

