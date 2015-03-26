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
  return isa::utils::toString(nrSamplesPerBlock);
}
std::string * getSNRDedispersedOpenCL(const snrDedispersedConf & conf, const std::string & dataType, const AstroData::Observation & observation) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void snrDedispersed(__global const " + dataType + " * const restrict dedispersedData, __global float * const restrict snrData) {\n"
    "unsigned int sample = get_local_id(0);\n"
    "unsigned int dm = get_group_id(1);\n"
    "__local float reductionCOU[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "__local " + dataType + " reductionMAX[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "__local float reductionMEA[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "__local float reductionVAR[" + isa::utils::toString(isa::utils::pad(conf.getNrSamplesPerBlock(), observation.getPadding())) + "];\n"
    "float counter = 0;\n"
    + dataType + " max = 0;\n"
    "float variance = 0.0f;\n"
    "float mean = 0.0f;\n"
    "\n"
    "// Compute phase"
    "for ( ; sample < " + isa::utils::toString(observation.getNrSamplesPerSecond()) + "; sample += " + isa::utils::toString(conf.getNrSamplesPerBlock()) + " ) {\n"
    + datatype + " item = dedispersedData[(dm * " + isa::utils::toString(observation.getNrSamplesPerPaddedSecond()) + ") + sample];\n"
    "counter += 1.0f;\n"
    "float delta = item - mean;\n"
    "max = fmax(max, item);\n"
    "mean += delta / counter;\n"
    "variance += delta * (item - mean);\n"
    "}\n"
    "reductionCOU[get_local_id(0)] = counter;\n"
    "reductionMAX[get_local_id(0)] = max;\n"
    "reductionMEA[get_local_id(0)] = mean;\n"
    "reductionVAR[get_local_id(0)] = variance / (counter - 1);\n"
    "barrier(CL_LOCAL_MEM_FENCE);\n"
    "// Reduce phase"
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
    "reductionVAR[get_local_id(0)] = variance / (counter - 1);\n"
    "}\n"
    "barrier(CL_LOCAL_MEM_FENCE);\n"
    "}\n"
    "// Store"
    "if ( get_local_id(0) == 0 ) {\n"
    "snrData[dm] = (max - mean) / native_sqrt(variance);\n"
    "}\n"
    "}\n";
  // End kernel's template

  return code;
}
} // PulsarSearch

