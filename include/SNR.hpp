// Copyright 2017 Netherlands Institute for Radio Astronomy (ASTRON)
// Copyright 2017 Netherlands eScience Center
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
#include <algorithm>

#include <Kernel.hpp>
#include <Observation.hpp>
#include <Platform.hpp>
#include <utils.hpp>

#pragma once

namespace SNR
{

/**
 ** @brief Configuration class.
 */
class snrConf : public isa::OpenCL::KernelConf
{
  public:
    snrConf();
    ~snrConf();
    // Get
    bool getSubbandDedispersion() const;
    // Set
    void setSubbandDedispersion(bool subband);
    // utils
    std::string print() const;

  private:
    bool subbandDedispersion;
};
typedef std::map<std::string, std::map<unsigned int, std::map<unsigned int, SNR::snrConf *> *> *> tunedSNRConf;

/**
 ** @brief Order of the underlying data.
 */
enum DataOrdering
{
    DMsSamples,
    SamplesDMs
};

/**
 ** @brief Kernel to test or tune.
 */
enum Kernel
{
    SNR,
    Max,
    MedianOfMedians,
    MedianOfMediansAbsoluteDeviation,
    AbsoluteDeviation
};

/**
 ** @brief Generate OpenCL code for the "max" kernel.
 ** The "max" operator is used to find, for all dedispersed time series, the element with highest intensity.
 */
template <typename DataType>
std::string *getMaxOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding);
template <typename DataType>
std::string *getMaxDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding);
/**
 ** @brief Generate OpenCL code for the median of medians kernel.
 */
template <typename DataType>
std::string *getMedianOfMediansOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding);
template <typename DataType>
std::string *getMedianOfMediansDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding);
/**
 ** @brief CPU control version of median of medians.
 */
template <typename DataType>
void medianOfMedians(const unsigned int stepSize, const std::vector<DataType> &timeSeries, std::vector<DataType> &medians, const AstroData::Observation &observation, const unsigned int padding);
/**
 ** @brief Generate OpenCL code for the median of medians absolute deviation kernel.
 */
template <typename DataType>
std::string *getMedianOfMediansAbsoluteDeviationOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding);
template <typename DataType>
std::string *getMedianOfMediansAbsoluteDeviationDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding);
/**
 ** @brief CPU control version of median of medians absolute deviation.
 */
template <typename DataType>
void medianOfMediansAbsoluteDeviation(const unsigned int stepSize, const std::vector<DataType> &baselines, const std::vector<DataType> &timeSeries, std::vector<DataType> &medians, const AstroData::Observation &observation, const unsigned int padding);
/**
 ** @brief Generate OpenCL code for for the absolute deviation kernel.
 */
template <typename DataType>
std::string * getAbsoluteDeviationOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding);
template <typename DataType>
std::string * getAbsoluteDeviationDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding);
/**
 ** @brief CPU control version of absolute deviation.
 */
template <typename DataType>
void absoluteDeviation(const DataType baseline, const std::vector<DataType> &timeSeries, std::vector<DataType> &absoluteDeviations, const AstroData::Observation &observation, const unsigned int padding);
// OpenCL SNR
template <typename T>
std::string *getSNRDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int nrSamples, const unsigned int padding);
template <typename T>
std::string *getSNRSamplesDMsOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int nrSamples, const unsigned int padding);
// Read configuration files
void readTunedSNRConf(tunedSNRConf &tunedSNR, const std::string &snrFilename);

// Implementations
inline bool snrConf::getSubbandDedispersion() const
{
    return subbandDedispersion;
}

inline void snrConf::setSubbandDedispersion(bool subband)
{
    subbandDedispersion = subband;
}

template <typename DataType>
std::string *getMaxOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding)
{
    std::string *code = 0;

    if (ordering == DataOrdering::DMsSamples)
    {
        code = getMaxDMsSamplesOpenCL<DataType>(conf, dataName, observation, downsampling, padding);
    }
    return code;
}

template <typename DataType>
std::string *getMaxDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding)
{
    std::string *code = new std::string();
    unsigned int nrSamples = 0;
    unsigned int nrDMs = 0;

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
        nrSamples = observation.getNrSamplesPerBatch() / downsampling;
    // Generate source code
    *code = "__kernel void getMax_DMsSamples_" + std::to_string(nrSamples) + "(__global const " + dataName + " * const restrict time_series, __global " + dataName + " * const restrict max_values, __global unsigned int * const restrict max_indices) {\n"
        "<%LOCAL_VARIABLES%>"
        "__local " + dataName + " reduction_value[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "__local unsigned int reduction_index[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "\n"
        "for ( unsigned int value_id = get_local_id(0) + " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + "; value_id < " + std::to_string(nrSamples) + "; value_id += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
        + dataName + " value;\n"
        "\n"
        "<%LOCAL_COMPUTE%>"
        "}\n"
        "<%LOCAL_REDUCE%>"
        "reduction_value[get_local_id(0)] = value_0;\n"
        "reduction_index[get_local_id(0)] = index_0;\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "unsigned int threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
        "for ( unsigned int value_id = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
        "if ( (value_id < threshold) && (reduction_value[value_id + threshold] > value_0) ) {\n"
        "value_0 = reduction_value[value_id + threshold];\n"
        "reduction_value[value_id] = value_0;\n"
        "index_0 = reduction_index[value_id + threshold];\n"
        "reduction_index[value_id] = index_0;\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"
        "if ( get_local_id(0) == 0 ) {\n"
        "max_values[(get_group_id(2) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(DataType))) + ") + get_group_id(1)] = value_0;\n"
        "max_indices[(get_group_id(2) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(unsigned int))) + ") + get_group_id(1)] = index_0;\n"
        "}\n"
        "}\n";
    std::string localVariablesTemplate = dataName + " value_<%ITEM_NUMBER%> = time_series[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + get_local_id(0) + <%ITEM_OFFSET%>];\n"
        "unsigned int index_<%ITEM_NUMBER%> = get_local_id(0) + <%ITEM_OFFSET%>;\n";
    std::string localComputeNoCheckTemplate = "value = time_series[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + value_id + <%ITEM_OFFSET%>];\n"
        "if ( value > value_<%ITEM_NUMBER%> ) {\n"
        "value_<%ITEM_NUMBER%> = value;\n"
        "index_<%ITEM_NUMBER%> = value_id + <%ITEM_OFFSET%>;\n"
        "}\n";
    std::string localComputeCheckTemplate = "if ( value_id + <%ITEM_OFFSET%> < " + std::to_string(nrSamples) + " ) {\n"
        "value = time_series[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + value_id + <%ITEM_OFFSET%>];\n"
        "if ( value > value_<%ITEM_NUMBER%> ) {\n"
        "value_<%ITEM_NUMBER%> = value;\n"
        "index_<%ITEM_NUMBER%> = value_id + <%ITEM_OFFSET%>;\n"
        "}\n"
        "}\n";
    std::string localReduceTemplate = "if ( value_<%ITEM_NUMBER%> > value_0 ) {\n"
        "value_0 = value_<%ITEM_NUMBER%>;\n"
        "index_0 = index_<%ITEM_NUMBER%>;\n"
        "}\n";
    std::string localVariables;
    std::string localCompute;
    std::string localReduce;
    for (unsigned int item = 0; item < conf.getNrItemsD0(); item++)
    {
        std::string *temp;
        std::string itemString = std::to_string(item);
        std::string itemOffsetString = std::to_string(item * conf.getNrThreadsD0());
        temp = isa::utils::replace(&localVariablesTemplate, "<%ITEM_NUMBER%>", itemString);
        if (item == 0)
        {
            temp = isa::utils::replace(temp, " + <%ITEM_OFFSET%>", std::string(), true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%ITEM_OFFSET%>", itemOffsetString, true);
        }
        localVariables.append(*temp);
        delete temp;
        if ((nrSamples % (conf.getNrThreadsD0() * conf.getNrItemsD0())) == 0)
        {
            temp = isa::utils::replace(&localComputeNoCheckTemplate, "<%ITEM_NUMBER%>", itemString);
        }
        else
        {
            temp = isa::utils::replace(&localComputeCheckTemplate, "<%ITEM_NUMBER%>", itemString);
        }
        if (item == 0)
        {
            temp = isa::utils::replace(temp, " + <%ITEM_OFFSET%>", std::string(), true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%ITEM_OFFSET%>", itemOffsetString, true);
        }
        localCompute.append(*temp);
        delete temp;
        if (item > 0)
        {
            temp = isa::utils::replace(&localReduceTemplate, "<%ITEM_NUMBER%>", itemString);
            localReduce.append(*temp);
            delete temp;
        }
    }
    code = isa::utils::replace(code, "<%LOCAL_VARIABLES%>", localVariables, true);
    code = isa::utils::replace(code, "<%LOCAL_COMPUTE%>", localCompute, true);
    code = isa::utils::replace(code, "<%LOCAL_REDUCE%>", localReduce, true);
    return code;
}

template <typename DataType>
std::string *getMedianOfMediansOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding)
{
    std::string *code = 0;

    if (ordering == DataOrdering::DMsSamples)
    {
        code = getMedianOfMediansDMsSamplesOpenCL<DataType>(conf, dataName, observation, downsampling, stepSize, padding);
    }
    return code;
}

template <typename DataType>
std::string *getMedianOfMediansDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding)
{
    std::string *code = new std::string();
    unsigned int nrSamples = 0;
    unsigned int nrDMs = 0;

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    nrSamples = observation.getNrSamplesPerBatch() / downsampling;
    // Generate source code
    *code = "__kernel void medianOfMedians_DMsSamples_" + std::to_string(stepSize) + "(__global const " + dataName + " * const restrict time_series, __global " + dataName + " * const restrict medians) {\n"
        "__local " + dataName + " local_data[" + std::to_string(stepSize) + "];\n"
        "\n"
        "// Load data in shared memory\n"
        "for ( unsigned int item = get_local_id(0); item < " + std::to_string(stepSize) + "; item += " + std::to_string(conf.getNrThreadsD0()) + " ) {\n"
        "local_data[item] = time_series[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(stepSize) + ") + item];\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "// Odd Even Sort\n"
        "for ( unsigned int item = 0; item < " + std::to_string(stepSize) + "; item++ ) {\n"
        "if ( (item % 2) == 1 ) {\n"
        "if ( (get_local_id(0) % 2) == 1 ) {\n"
        "if ( local_data[item] > local_data[item + 1] ) {\n"
        "" + dataName + " temp = local_data[item];\n"
        "local_data[item] = local_data[item + 1];\n"
        "local_data[item + 1] = temp;\n"
        "}\n"
        "} else {\n"
        "if ( local_data[item] < local_data[item - 1] ) {\n"
        "" + dataName + " temp = local_data[item];\n"
        "local_data[item] = local_data[item - 1];\n"
        "local_data[item - 1] = temp;\n"
        "}\n"
        "}\n"
        "} else {\n"
        "if ( (get_local_id(0) % 2) == 0 ) {\n"
        "if ( local_data[item] > local_data[item + 1] ) {\n"
        "" + dataName + " temp = local_data[item];\n"
        "local_data[item] = local_data[item + 1];\n"
        "local_data[item + 1] = temp;\n"
        "}\n"
        "} else {\n"
        "if ( local_data[item] < local_data[item - 1] ) {\n"
        "" + dataName + " temp = local_data[item];\n"
        "local_data[item] = local_data[item - 1];\n"
        "local_data[item - 1] = temp;\n"
        "}\n"
        "}\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"
        "// Store median\n"
        "if ( get_local_id(0) == 0 ) {\n"
        "<%STORE%>"
        "}\n"
        "}\n";
    std::string storeTemplateFirstStep = "medians[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples / stepSize, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples / stepSize, padding / sizeof(DataType))) + ") + get_group_id(0)] = local_data[" + std::to_string(stepSize / 2) + "];\n";
    std::string storeTemplateSecondStep = "medians[(get_group_id(2) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(DataType))) + ") + get_group_id(1)] = local_data[" + std::to_string(stepSize / 2) + "];\n";
    if (nrSamples != stepSize)
    {
        code = isa::utils::replace(code, "<%STORE%>", storeTemplateFirstStep, true);
    }
    else
    {
        code = isa::utils::replace(code, "<%STORE%>", storeTemplateSecondStep, true);
    }
    return code;
}

template <typename DataType>
void medianOfMedians(const unsigned int stepSize, const std::vector<DataType> &timeSeries, std::vector<DataType> &medians, const AstroData::Observation &observation, const unsigned int padding)
{
    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
        {
            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
            {
                for (unsigned int step = 0; step < observation.getNrSamplesPerBatch() / stepSize; step++)
                {
                    std::vector<DataType> localArray;

                    for (unsigned int sample = 0; sample < stepSize; sample++)
                    {
                        localArray.push_back(timeSeries.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (step * stepSize) + sample));
                    }
                    std::sort(localArray.begin(), localArray.end());
                    if (stepSize == observation.getNrSamplesPerBatch())
                    {
                        medians.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs()) + dm) = localArray.at(stepSize / 2);
                    }
                    else
                    {
                        medians.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / stepSize, padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / stepSize, padding / sizeof(DataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / stepSize, padding / sizeof(DataType))) + step) = localArray.at(stepSize / 2);
                    }
                }
            }
        }
    }
}

template <typename DataType>
std::string *getMedianOfMediansAbsoluteDeviationOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding)
{
    std::string *code = 0;

    if (ordering == DataOrdering::DMsSamples)
    {
        code = getMedianOfMediansAbsoluteDeviationDMsSamplesOpenCL<DataType>(conf, dataName, observation, downsampling, stepSize, padding);
    }
    return code;
}

template <typename DataType>
std::string *getMedianOfMediansAbsoluteDeviationDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int stepSize, const unsigned int padding)
{
    std::string *code = new std::string();
    unsigned int nrSamples = 0;
    unsigned int nrDMs = 0;

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    nrSamples = observation.getNrSamplesPerBatch() / downsampling;
    // Generate source code
    *code = "__kernel void medianOfMediansAbsoluteDeviation_DMsSamples_" + std::to_string(stepSize) + "(__global const " + dataName + " * const restrict baselines, __global const " + dataName + " * const restrict time_series, __global " + dataName + " * const restrict medians) {\n"
        "__local " + dataName + " local_data[" + std::to_string(stepSize) + "];\n"
        + dataName + " baseline = baselines[(get_group_id(2) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(DataType))) + ") + get_group_id(1)];\n"
        "\n"
        "// Load data in shared memory\n"
        "for ( unsigned int item = get_local_id(0); item < " + std::to_string(stepSize) + "; item += " + std::to_string(conf.getNrThreadsD0()) + " ) {\n"
        "local_data[item] = fabs(time_series[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(stepSize) + ") + item] - baseline);\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "// Bubble Sort\n"
        "for ( unsigned int step = 0; step < " + std::to_string(stepSize) + "; step++ ) {\n"
        "if ( (get_local_id(0) % 2) == (step % 2) ) {\n"
        "for ( unsigned int item = get_local_id(0); item < " + std::to_string(stepSize) + " - 1; item += " + std::to_string(conf.getNrThreadsD0()) + " ) {\n"
        "if ( local_data[item] > local_data[item + 1] ) {\n"
        "" + dataName + " temp = local_data[item];\n"
        "local_data[item] = local_data[item + 1];\n"
        "local_data[item + 1] = temp;\n"
        "}\n"
        "}\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"
        "// Store median\n"
        "if ( get_local_id(0) == 0 ) {\n"
        "medians[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples / stepSize, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples / stepSize, padding / sizeof(DataType))) + ") + get_group_id(0)] = local_data[" + std::to_string(stepSize / 2) + "];\n"
        "}\n"
        "}\n";
    return code;
}

template <typename DataType>
void medianOfMediansAbsoluteDeviation(const unsigned int stepSize, const std::vector<DataType> &baselines, const std::vector<DataType> &timeSeries, std::vector<DataType> &medians, const AstroData::Observation &observation, const unsigned int padding)
{
    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
        {
            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
            {
                for (unsigned int step = 0; step < observation.getNrSamplesPerBatch() / stepSize; step++)
                {
                    std::vector<DataType> localArray;

                    for (unsigned int sample = 0; sample < stepSize; sample++)
                    {
                        localArray.push_back(std::abs(timeSeries.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (step * stepSize) + sample) - baselines.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs()) + dm)));
                    }
                    std::sort(localArray.begin(), localArray.end());
                    medians.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / stepSize, padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / stepSize, padding / sizeof(DataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / stepSize, padding / sizeof(DataType))) + step) = localArray.at(stepSize / 2);
                }
            }
        }
    }
}

template <typename DataType>
std::string * getAbsoluteDeviationOpenCL(const snrConf &conf, const DataOrdering ordering, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding)
{
    std::string *code = 0;

    if (ordering == DataOrdering::DMsSamples)
    {
        code = getAbsoluteDeviationDMsSamplesOpenCL<DataType>(conf, dataName, observation, downsampling, padding);
    }
    return code;
}

template <typename DataType>
std::string * getAbsoluteDeviationDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int downsampling, const unsigned int padding)
{
    std::string *code = new std::string();
    unsigned int nrSamples = 0;
    unsigned int nrDMs = 0;

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    nrSamples = observation.getNrSamplesPerBatch() / downsampling;
    // Generate source code
    *code = "__kernel void absolute_deviation_DMsSamples_" + std::to_string(nrSamples) + "(__global const " + dataName + " * const restrict baselines, __global const " + dataName + " * const restrict input_data, __global " + dataName + " * const restrict output_data) {\n"
        "unsigned int item = (get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(DataType))) + ") + (get_group_id(0) * " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
        "<%COMPUTE_STORE%>"
        "}\n";
    std::string computeStoreTemplate = "output_data[item + <%ITEM_OFFSET%>] = fabs(input_data[item + <%ITEM_OFFSET%>] - baselines[(get_group_id(2) * " + std::to_string(nrDMs * (padding / sizeof(DataType))) + ") + get_group_id(1)]);\n";
    std::string computeStore;
    for (unsigned int item = 0; item < conf.getNrItemsD0(); item++)
    {
        std::string *temp;
        std::string itemOffsetString = std::to_string(item * conf.getNrThreadsD0());
        if (item == 0)
        {
            temp = isa::utils::replace(&computeStoreTemplate, " + <%ITEM_OFFSET%>", std::string());
        }
        else
        {
            temp = isa::utils::replace(&computeStoreTemplate, "<%ITEM_OFFSET%>", itemOffsetString);
        }
        computeStore.append(*temp);
        delete temp;
    }
    code = isa::utils::replace(code, "<%COMPUTE_STORE%>", computeStore, true);
    return code;
}

template <typename DataType>
void absoluteDeviation(const std::vector<DataType> &baselines, const std::vector<DataType> &timeSeries, std::vector<DataType> &absoluteDeviations, const AstroData::Observation &observation, const unsigned int padding)
{
    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
        {
            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
            {
                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                {
                    absoluteDeviations.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + sample) = std::abs(timeSeries.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(DataType))) + sample) - baselines.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * (padding / sizeof(DataType))) + (subbandingDM * observation.getNrDMs()) + dm));
                }
            }
        }
    }
}

template <typename T>
std::string *getSNRDMsSamplesOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int nrSamples, const unsigned int padding)
{
    std::string *code = new std::string();
    unsigned int nrDMs = 0;
    
    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    *code = "__kernel void snrDMsSamples" + std::to_string(nrSamples) + "(__global const " + dataName + " * const restrict input, __global float * const restrict outputSNR, __global unsigned int * const restrict outputSample) {\n"
        "float delta = 0.0f;\n"
        "<%DEF%>"
        "__local float reductionCOU[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "__local " + dataName + " reductionMAX[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "__local unsigned int reductionSAM[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "__local float reductionMEA[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "__local float reductionVAR[" + std::to_string(conf.getNrThreadsD0()) + "];\n"
        "\n"
        "// Compute phase\n"
        "for ( unsigned int sample = get_local_id(0) + " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + "; sample < " + std::to_string(nrSamples) + "; sample += " + std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + " ) {\n"
        + dataName + " item = 0;\n"
        "<%COMPUTE%>"
        "}\n"
        "// In-thread reduce\n"
        "<%REDUCE%>"
        "// Local memory store\n"
        "reductionCOU[get_local_id(0)] = counter0;\n"
        "reductionMAX[get_local_id(0)] = max0;\n"
        "reductionSAM[get_local_id(0)] = maxSample0;\n"
        "reductionMEA[get_local_id(0)] = mean0;\n"
        "reductionVAR[get_local_id(0)] = variance0;\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "// Reduce phase\n"
        "unsigned int threshold = " + std::to_string(conf.getNrThreadsD0() / 2) + ";\n"
        "for ( unsigned int sample = get_local_id(0); threshold > 0; threshold /= 2 ) {\n"
        "if ( sample < threshold ) {\n"
        "delta = reductionMEA[sample + threshold] - mean0;\n"
        "counter0 += reductionCOU[sample + threshold];\n"
        "mean0 = ((reductionCOU[sample] * mean0) + (reductionCOU[sample + threshold] * reductionMEA[sample + threshold])) / counter0;\n"
        "variance0 += reductionVAR[sample + threshold] + ((delta * delta) * ((reductionCOU[sample] * reductionCOU[sample + threshold]) / counter0));\n"
        "if ( reductionMAX[sample + threshold] > max0 ) {\n"
        "max0 = reductionMAX[sample + threshold];\n"
        "maxSample0 = reductionSAM[sample + threshold];\n"
        "}\n"
        "reductionCOU[sample] = counter0;\n"
        "reductionMAX[sample] = max0;\n"
        "reductionSAM[sample] = maxSample0;\n"
        "reductionMEA[sample] = mean0;\n"
        "reductionVAR[sample] = variance0;\n"
        "}\n"
        "barrier(CLK_LOCAL_MEM_FENCE);\n"
        "}\n"
        "// Store\n"
        "if ( get_local_id(0) == 0 ) {\n"
        "outputSNR[(get_group_id(2) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(float))) + ") + get_group_id(1)] = (max0 - mean0) / native_sqrt(variance0 * " + std::to_string(1.0f / (nrSamples - 1)) + "f);\n"
        "outputSample[(get_group_id(2) * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(unsigned int))) + ") + get_group_id(1)] = maxSample0;\n"
        "}\n"
        "}\n";
    std::string def_sTemplate = dataName + " max<%NUM%> = input[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (get_local_id(0) + <%OFFSET%>)];\n"
        "unsigned int maxSample<%NUM%> = get_local_id(0) + <%OFFSET%>;\n"
        "float counter<%NUM%> = 1.0f;\n"
        "float variance<%NUM%> = 0.0f;\n"
        "float mean<%NUM%> = max<%NUM%>;\n";
    std::string compute_sTemplate;
    if ((nrSamples % (conf.getNrThreadsD0() * conf.getNrItemsD0())) != 0)
    {
        compute_sTemplate += "if ( (sample + <%OFFSET%>) < " + std::to_string(nrSamples) + " ) {\n";
    }
    compute_sTemplate += "item = input[(get_group_id(2) * " + std::to_string(nrDMs * isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (get_group_id(1) * " + std::to_string(isa::utils::pad(nrSamples, padding / sizeof(T))) + ") + (sample + <%OFFSET%>)];\n"
        "counter<%NUM%> += 1.0f;\n"
        "delta = item - mean<%NUM%>;\n"
        "mean<%NUM%> += delta / counter<%NUM%>;\n"
        "variance<%NUM%> += delta * (item - mean<%NUM%>);\n"
        "if ( item > max<%NUM%> ) {\n"
        "max<%NUM%> = item;\n"
        "maxSample<%NUM%> = sample + <%OFFSET%>;\n"
        "}\n";
    if ((nrSamples % (conf.getNrThreadsD0() * conf.getNrItemsD0())) != 0)
    {
        compute_sTemplate += "}\n";
    }
    std::string reduce_sTemplate = "delta = mean<%NUM%> - mean0;\n"
        "counter0 += counter<%NUM%>;\n"
        "mean0 = (((counter0 - counter<%NUM%>) * mean0) + (counter<%NUM%> * mean<%NUM%>)) / counter0;\n"
        "variance0 += variance<%NUM%> + ((delta * delta) * (((counter0 - counter<%NUM%>) * counter<%NUM%>) / counter0));\n"
        "if ( max<%NUM%> > max0 ) {\n"
        "max0 = max<%NUM%>;\n"
        "maxSample0 = maxSample<%NUM%>;\n"
        "}\n";

    std::string *def_s = new std::string();
    std::string *compute_s = new std::string();
    std::string *reduce_s = new std::string();

    for (unsigned int sample = 0; sample < conf.getNrItemsD0(); sample++)
    {
        std::string sample_s = std::to_string(sample);
        std::string offset_s = std::to_string(conf.getNrThreadsD0() * sample);
        std::string *temp = 0;

        temp = isa::utils::replace(&def_sTemplate, "<%NUM%>", sample_s);
        if (sample == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        def_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&compute_sTemplate, "<%NUM%>", sample_s);
        if (sample == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        compute_s->append(*temp);
        delete temp;
        if (sample == 0)
        {
            continue;
        }
        temp = isa::utils::replace(&reduce_sTemplate, "<%NUM%>", sample_s);
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

template <typename T>
std::string *getSNRSamplesDMsOpenCL(const snrConf &conf, const std::string &dataName, const AstroData::Observation &observation, const unsigned int nrSamples, const unsigned int padding)
{
    unsigned int nrDMs = 0;
    std::string *code = new std::string();

    if (conf.getSubbandDedispersion())
    {
        nrDMs = observation.getNrDMs(true) * observation.getNrDMs();
    }
    else
    {
        nrDMs = observation.getNrDMs();
    }
    // Begin kernel's template
    *code = "__kernel void snrSamplesDMs" + std::to_string(nrDMs) + "(__global const " + dataName + " * const restrict input, __global float * const restrict outputSNR, __global unsigned int * const restrict outputSample) {\n"
                                                                                                    "unsigned int dm = (get_group_id(0) * " +
            std::to_string(conf.getNrThreadsD0() * conf.getNrItemsD0()) + ") + get_local_id(0);\n"
                                                                          "float delta = 0.0f;\n"
                                                                          "<%DEF%>"
                                                                          "\n"
                                                                          "for ( unsigned int sample = 1; sample < " +
            std::to_string(nrSamples) + "; sample++ ) {\n" + dataName + " item = 0;\n"
                                                                        "<%COMPUTE%>"
                                                                        "}\n"
                                                                        "<%STORE%>"
                                                                        "}\n";
    std::string def_sTemplate = "float counter<%NUM%> = 1.0f;\n" + dataName + " max<%NUM%> = input[get_group_id(1) + <%OFFSET%>];\n"
                                                                              "unsigned int maxSample<%NUM%> = 0;\n"
                                                                              "float variance<%NUM%> = 0.0f;\n"
                                                                              "float mean<%NUM%> = max<%NUM%>;\n";
    std::string compute_sTemplate = "item = input[(beam * " + std::to_string(nrSamples * isa::utils::pad(nrDMs, padding / sizeof(T))) + ") + (sample * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(T))) + ")  + (dm + <%OFFSET%>)];\n"
                                                                                                                                                                                                                         "counter<%NUM%> += 1.0f;\n"
                                                                                                                                                                                                                         "delta = item - mean<%NUM%>;\n"
                                                                                                                                                                                                                         "mean<%NUM%> += delta / counter<%NUM%>;\n"
                                                                                                                                                                                                                         "variance<%NUM%> += delta * (item - mean<%NUM%>);\n"
                                                                                                                                                                                                                         "if ( item > max<%NUM%> ) {\n"
                                                                                                                                                                                                                         "max<%NUM%> = item;\n"
                                                                                                                                                                                                                         "maxSample<%NUM%> = sample;\n"
                                                                                                                                                                                                                         "}\n";
    std::string store_sTemplate = "outputSNR[(beam * " + std::to_string(isa::utils::pad(nrDMs, padding / sizeof(float))) + ") + dm + <%OFFSET%>] = (max<%NUM%> - mean<%NUM%>) / native_sqrt(variance<%NUM%> * " + std::to_string(1.0f / (observation.getNrSamplesPerBatch() - 1)) + "f);\n";
    // End kernel's template

    std::string *def_s = new std::string();
    std::string *compute_s = new std::string();
    std::string *store_s = new std::string();

    for (unsigned int dm = 0; dm < conf.getNrItemsD0(); dm++)
    {
        std::string dm_s = std::to_string(dm);
        std::string offset_s = std::to_string(conf.getNrThreadsD0() * dm);
        std::string *temp = 0;

        temp = isa::utils::replace(&def_sTemplate, "<%NUM%>", dm_s);
        if (dm == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        def_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&compute_sTemplate, "<%NUM%>", dm_s);
        if (dm == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        compute_s->append(*temp);
        delete temp;
        temp = isa::utils::replace(&store_sTemplate, "<%NUM%>", dm_s);
        if (dm == 0)
        {
            std::string empty_s("");
            temp = isa::utils::replace(temp, " + <%OFFSET%>", empty_s, true);
        }
        else
        {
            temp = isa::utils::replace(temp, "<%OFFSET%>", offset_s, true);
        }
        store_s->append(*temp);
        delete temp;
    }

    code = isa::utils::replace(code, "<%DEF%>", *def_s, true);
    code = isa::utils::replace(code, "<%COMPUTE%>", *compute_s, true);
    code = isa::utils::replace(code, "<%STORE%>", *store_s, true);
    delete def_s;
    delete compute_s;
    delete store_s;

    return code;
}

} // SNR
