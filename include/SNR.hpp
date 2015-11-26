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


#ifndef SNR_HPP
#define SNR_HPP

namespace PulsarSearch {

class snrConf {
public:
  snrConf();
  ~snrConf();
  // Get
  unsigned int getNrThreadsD0() const;
  unsigned int getNrItemsD0() const;
  // Set
  void setNrThreadsD0(unsigned int threads);
  void setNrItemsD0(unsigned int items);
  // utils
  std::string print() const;

private:
  unsigned int nrThreadsD0;
  unsigned int nrItemsD0;
};

typedef std::map< std::string, std::map< unsigned int, std::map< unsigned int, PulsarSearch::snrConf * > * > * > tunedSNRConf;

// OpenCL SNR
template< typename T > std::string * getSNRDMsSamplesOpenCL(const snrConf & conf, const std::string & dataType, const unsigned int nrSamples, const unsigned int padding);
// Read configuration files
void readTunedSNRConf(tunedSNRConf & tunedSNR, const std::string & snrFilename);


// Implementations
inline unsigned int snrConf::getNrThreadsD0() const {
  return nrThreadsD0;
}

inline unsigned int snrConf::getNrItemsD0() const {
  return nrItemsD0;
}

inline void snrConf::setNrThreadsD0(unsigned int threads) {
  nrThreadsD0 = threads;
}

inline void snrConf::setNrItemsD0(unsigned int items) {
  nrItemsD0 = items;
}

template< typename T > std::string * getSNRDMsSamplesOpenCL(const snrConf & conf, const std::string & dataType, const unsigned int nrSamples, const unsigned int padding) {
  std::string * code = new std::string();

  // Begin kernel's template
  *code = "__kernel void snrDMsSamples" + isa::utils::toString(nrSamples) + "(__global const " + dataType + " * const restrict input, __global float * const restrict output) {\n"
    "unsigned int dm = get_group_id(1);\n"
    "float delta = 0.0f;\n"
    "__local float reductionCOU[" + isa::utils::toString(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(T))) + "];\n"
    "__local " + dataType + " reductionMAX[" + isa::utils::toString(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(T))) + "];\n"
    "__local float reductionMEA[" + isa::utils::toString(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(T))) + "];\n"
    "__local float reductionVAR[" + isa::utils::toString(isa::utils::pad(conf.getNrThreadsD0(), padding / sizeof(T))) + "];\n"
    "<%DEF%>"
    "\n"
    "// Compute phase\n"
    "for ( unsigned int sample = get_local_id(0) + " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + "; sample < " + isa::utils::toString(nrSamples) + "; sample += " + isa::utils::toString(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
    + dataType + " item = 0;\n"
    "<%COMPUTE%>"
    "}\n"
    "// In-thread reduce\n"
    "<%REDUCE%>"
    "// Local memory store\n"
    "reductionCOU[get_local_id(0)] = counter0;\n"
    "reductionMAX[get_local_id(0)] = max0;\n"
    "reductionMEA[get_local_id(0)] = mean0;\n"
    "reductionVAR[get_local_id(0)] = variance0;\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "// Reduce phase\n"
    "unsigned int threshold = " + isa::utils::toString(conf.getNrThreadsD0() / 2) + ";\n"
    "for ( unsigned int sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
    "if ( sample < threshold ) {\n"
    "delta = reductionMEA[sample + threshold] - mean0;\n"
    "counter0 += reductionCOU[sample + threshold];\n"
    "max0 = fmax(max0, reductionMAX[sample + threshold]);\n"
    "mean0 = ((reductionCOU[sample] * mean0) + (reductionCOU[sample + threshold] * reductionMEA[sample + threshold])) / counter0;\n"
    "variance0 += reductionVAR[sample + threshold] + ((delta * delta) * ((reductionCOU[sample] * reductionCOU[sample + threshold]) / counter0));\n"
    "reductionCOU[get_local_id(0)] = counter0;\n"
    "reductionMAX[get_local_id(0)] = max0;\n"
    "reductionMEA[get_local_id(0)] = mean0;\n"
    "reductionVAR[get_local_id(0)] = variance0;\n"
    "}\n"
    "barrier(CLK_LOCAL_MEM_FENCE);\n"
    "}\n"
    "// Store\n"
    "if ( get_local_id(0) == 0 ) {\n"
    "output[dm] = (max0 - mean0) / native_sqrt(variance0 * " + isa::utils::toString(1.0f / (nrSamples - 1)) + "f);\n"
    "}\n"
    "}\n";
  std::string defTemplate = "float counter<%NUM%> = 1.0f;\n"
    + dataType + " max<%NUM%> = input[(dm * " + isa::utils::toString(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (get_local_id(0) + <%OFFSET%>)];\n"
    "float variance<%NUM%> = 0.0f;\n"
    "float mean<%NUM%> = max<%NUM%>;\n";
  std::string computeTemplate = "item = input[(dm * " + isa::utils::toString(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (sample + <%OFFSET%>)];\n"
    "counter<%NUM%> += 1.0f;\n"
    "delta = item - mean<%NUM%>;\n"
    "max<%NUM%> = fmax(max<%NUM%>, item);\n"
    "mean<%NUM%> += delta / counter<%NUM%>;\n"
    "variance<%NUM%> += delta * (item - mean<%NUM%>);\n";
  std::string reduceTemplate = "delta = mean<%NUM%> - mean0;\n"
    "counter0 += counter<%NUM%>;\n"
    "max0 = fmax(max0, max<%NUM%>);\n"
    "mean0 = (((counter0 - counter<%NUM%>) * mean0) + (counter<%NUM%> * mean<%NUM%>)) / counter0;\n"
    "variance0 += variance<%NUM%> + ((delta * delta) * (((counter0 - counter<%NUM%>) * counter<%NUM%>) / counter0));\n";
  // End kernel's template

  std::string * def_s = new std::string();
  std::string * compute_s = new std::string();
  std::string * reduce_s = new std::string();

  for ( unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++ ) {
    std::string sample_s = isa::utils::toString(sample);
    std::string offset_s = isa::utils::toString(conf.getNrThreadsD0() * sample);
    std::string * temp = 0;

    temp = isa::utils::replace(&defTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    def_s->append(*temp);
    delete temp;
    temp = isa::utils::replace(&computeTemplate, "<%NUM%>", sample_s);
    if ( sample == 0 ) {
      std::string empty_s("");
      temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
    } else {
      temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
    }
    compute_s->append(*temp);
    delete temp;
    if ( sample == 0 ) {
      continue;
    }
    temp = isa::utils::replace(&reduceTemplate, "<%NUM%>", sample_s);
    reduce_s->append(*temp);
    delete temp;
  }

  code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
  code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
  code = isa::utils::replace(code, "<%REDUCE%>", *reduce_s, true);
  delete def_s;
  delete compute_s;
  delete reduce_s;

  return code;
}

} // PulsarSearch

#endif // SNR_HPP

