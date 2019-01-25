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
#include <Statistics.hpp>

int test(const bool printResults, const bool printCode, const unsigned int clPlatformID, const unsigned int clDeviceID, const SNR::DataOrdering ordering, const SNR::Kernel kernelUnderTest, const unsigned int padding, const AstroData::Observation &observation, const SNR::snrConf &conf, const unsigned int medianStep = 0, const float nSigma = 3.0f);

int main(int argc, char *argv[])
{
    bool printCode = false;
    bool printResults = false;
    int returnCode = 0;
    unsigned int padding = 0;
    unsigned int clPlatformID = 0;
    unsigned int clDeviceID = 0;
    unsigned int stepSize = 0;
    float nSigma;
    SNR::Kernel kernel;
    SNR::DataOrdering ordering;
    AstroData::Observation observation;
    SNR::snrConf conf;

    try
    {
        isa::utils::ArgumentList args(argc, argv);
        if (args.getSwitch("-snr"))
        {
            kernel = SNR::Kernel::SNR;
        }
        else if (args.getSwitch("-max"))
        {
            kernel = SNR::Kernel::Max;
        }
        else if (args.getSwitch("-max_std"))
        {
            kernel = SNR::Kernel::MaxStdSigmaCut;
        }
        else if (args.getSwitch("-median"))
        {
            kernel = SNR::Kernel::MedianOfMedians;
        }
        else if (args.getSwitch("-momad"))
        {
            kernel = SNR::Kernel::MedianOfMediansAbsoluteDeviation;
        }
        else if (args.getSwitch("-absolute_deviation"))
        {
            kernel = SNR::Kernel::AbsoluteDeviation;
        }
        else{
            std::cerr << "One switch between -snr -max -max_std -median -momad and -absolute_deviation is required." << std::endl;
            return 1;
        }
        if (args.getSwitch("-dms_samples"))
        {
            ordering = SNR::DataOrdering::DMsSamples;
        }
        else if (args.getSwitch("-samples_dms"))
        {
            ordering = SNR::DataOrdering::SamplesDMs;
        }
        else
        {
            std::cerr << "One switch between -dms_samples and -samples_dms is required." << std::endl;
            return 1;
        }
        printCode = args.getSwitch("-print_code");
        printResults = args.getSwitch("-print_results");
        clPlatformID = args.getSwitchArgument<unsigned int>("-opencl_platform");
        clDeviceID = args.getSwitchArgument<unsigned int>("-opencl_device");
        padding = args.getSwitchArgument<unsigned int>("-padding");
        nSigma = args.getSwitchArgument<float>("-nsigma");
        conf.setNrThreadsD0(args.getSwitchArgument<unsigned int>("-threadsD0"));
        if ((kernel == SNR::Kernel::SNR) || (kernel == SNR::Kernel::Max) || (kernel == SNR::Kernel::AbsoluteDeviation))
        {
            conf.setNrItemsD0(args.getSwitchArgument<unsigned int>("-itemsD0"));
        }
        conf.setSubbandDedispersion(args.getSwitch("-subband"));
        observation.setNrSynthesizedBeams(args.getSwitchArgument<unsigned int>("-beams"));
        observation.setNrSamplesPerBatch(args.getSwitchArgument<unsigned int>("-samples"));
        if (conf.getSubbandDedispersion())
        {
            observation.setDMRange(args.getSwitchArgument<unsigned int>("-subbanding_dms"), 0.0f, 0.0f, true);
        }
        else
        {
            observation.setDMRange(1, 0.0f, 0.0f, true);
        }
        observation.setDMRange(args.getSwitchArgument<unsigned int>("-dms"), 0.0, 0.0);
        if (kernel == SNR::Kernel::MedianOfMedians || kernel == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
        {
            stepSize = args.getSwitchArgument<unsigned int>("-median_step");
        }
    }
    catch (isa::utils::SwitchNotFound &err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    catch (std::exception &err)
    {
        std::cerr << "Usage: " << argv[0] << " [-snr | -max | -max_std | -median | -momad | -absolute_deviation] [-dms_samples | -samples_dms] [-print_code] [-print_results] -opencl_platform ... -opencl_device ... -padding ... -threadsD0 ... -itemsD0 ... [-subband] -beams ... -dms ... -samples ..." << std::endl;
        std::cerr << "\t -subband -subbanding_dms ..." << std::endl;
        std::cerr << "\t -median -median_step ..." << std::endl;
        std::cerr << "\t -momad -median_step ..." << std::endl;
        std::cerr << "\t -max_std -nsigma ..." << std::endl;
        return 1;
    }
    if (kernel == SNR::Kernel::SNR || kernel == SNR::Kernel::Max || kernel == SNR::Kernel::MaxStdSigmaCut || kernel == SNR::Kernel::AbsoluteDeviation)
    {
        returnCode = test(printResults, printCode, clPlatformID, clDeviceID, ordering, kernel, padding, observation, conf);
    }
    else if (kernel == SNR::Kernel::MedianOfMedians || kernel == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
    {
        returnCode = test(printResults, printCode, clPlatformID, clDeviceID, ordering, kernel, padding, observation, conf, stepSize);
    }
    else if ( kernel == SNR::Kernel::MaxStdSigmaCut )
    {
        returnCode = test(printResults, printCode, clPlatformID, clDeviceID, ordering, kernel, padding, observation, conf, 0, nSigma);
    }

    return returnCode;
}

int test(const bool printResults, const bool printCode, const unsigned int clPlatformID, const unsigned int clDeviceID, const SNR::DataOrdering ordering, const SNR::Kernel kernelUnderTest, const unsigned int padding, const AstroData::Observation &observation, const SNR::snrConf &conf, const unsigned int medianStep, const float nSigma)
{
    uint64_t wrongSamples = 0;
    uint64_t wrongPositions = 0;
    uint64_t wrongSamples_stdev = 0;

    // Initialize OpenCL
    cl::Context *clContext = new cl::Context();
    std::vector<cl::Platform> *clPlatforms = new std::vector<cl::Platform>();
    std::vector<cl::Device> *clDevices = new std::vector<cl::Device>();
    std::vector<std::vector<cl::CommandQueue>> *clQueues = new std::vector<std::vector<cl::CommandQueue>>();
    isa::OpenCL::initializeOpenCL(clPlatformID, 1, clPlatforms, clContext, clDevices, clQueues);

    // Allocate memory
    std::vector<inputDataType> input;
    std::vector<outputDataType> output;
    std::vector<unsigned int> outputIndex;
    std::vector<outputDataType> baselines;
    std::vector<outputDataType> stdevs;
    cl::Buffer input_d, output_d, outputIndex_d, baselines_d, stdevs_d;

    if (ordering == SNR::DataOrdering::DMsSamples)
    {
        input.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType)));
    }
    else
    {
        input.resize(observation.getNrSynthesizedBeams() * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType)));
    }
    if (kernelUnderTest == SNR::Kernel::SNR || kernelUnderTest == SNR::Kernel::Max)
    {
        output.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float)));
        outputIndex.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int)));
    }
    if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
    {
        output.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float)));
        stdevs.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float)));
        outputIndex.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int)));
    }
    else if (kernelUnderTest == SNR::Kernel::MedianOfMedians)
    {
        if (medianStep != observation.getNrSamplesPerBatch())
        {
            output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType)));
        }
        else{
            output.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType)));
        }
    }
    else if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
    {
        output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType)));
        baselines.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType)));
    }
    else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
    {
        output.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType)));
        baselines.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType)));
    }
    try
    {
        input_d = cl::Buffer(*clContext, CL_MEM_READ_WRITE, input.size() * sizeof(inputDataType), 0, 0);
        output_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(outputDataType), 0, 0);
        stdevs_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, output.size() * sizeof(outputDataType), 0, 0);
        if (kernelUnderTest == SNR::Kernel::SNR || kernelUnderTest == SNR::Kernel::Max || kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            outputIndex_d = cl::Buffer(*clContext, CL_MEM_WRITE_ONLY, outputIndex.size() * sizeof(unsigned int), 0, 0);
        }
        if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation || kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
        {
            baselines_d = cl::Buffer(*clContext, CL_MEM_READ_ONLY, baselines.size() * sizeof(outputDataType), 0, 0);
        }
    }
    catch (cl::Error &err)
    {
        std::cerr << "OpenCL error allocating memory: " << std::to_string(err.err()) << "." << std::endl;
        return 1;
    }

    // Generate test data
    std::vector<unsigned int> maxSample(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int)));

    srand(time(0));
    for (auto item = maxSample.begin(); item != maxSample.end(); ++item)
    {
        *item = rand() % observation.getNrSamplesPerBatch();
    }
    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        if (printResults)
        {
            std::cout << "Beam: " << beam << std::endl;
        }
        if (ordering == SNR::DataOrdering::DMsSamples)
        {
            for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
            {
                for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                {
                    if (printResults)
                    {
                        std::cout << "Input -- DM: " << (subbandingDM * observation.getNrDMs()) + dm << " -- ";
                    }
                    for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                    {
                        if (sample == maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm))
                        {
                            input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample] = static_cast<inputDataType>(10 + (rand() % 10));
                        }
                        else
                        {
                            input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample] = static_cast<inputDataType>(rand() % 10);
                        }
                        if (printResults)
                        {
                            std::cout << input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample] << " ";
                        }
                    }
                    if (printResults)
                    {
                        std::cout << std::endl;
                    }
                }
            }
        }
        else
        {
            for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
            {
                if (printResults)
                {
                    std::cout << "Input -- Sample: " << sample << " -- ";
                }
                for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                {
                    for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                    {
                        if (sample == maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm))
                        {
                            input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm] = static_cast<inputDataType>(10 + (rand() % 10));
                        }
                        else
                        {
                            input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm] = static_cast<inputDataType>(rand() % 10);
                        }
                        if (printResults)
                        {
                            std::cout << input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm] << " ";
                        }
                    }
                    if (printResults)
                    {
                        std::cout << std::endl;
                    }
                }
            }
        }
    }
    if (printResults)
    {
        std::cout << std::endl;
    }
    if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation || kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
    {
        for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
        {
            if (printResults)
            {
                std::cout << "Baselines -- Beam: " << beam << " -- ";
            }
            for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
            {
                for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                {
                    baselines.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm) = static_cast<outputDataType>((std::rand() % 10) + 1);
                    if (printResults)
                    {
                        std::cout << baselines.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm) << " ";
                    }
                }
            }
            if (printResults)
            {
                std::cout << std::endl;
            }
        }
        if (printResults)
        {
            std::cout << std::endl;
        }
    }

    // Copy data structures to device
    try
    {
        clQueues->at(clDeviceID)[0].enqueueWriteBuffer(input_d, CL_FALSE, 0, input.size() * sizeof(inputDataType), reinterpret_cast<void *>(input.data()));
        if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation || kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
        {
            clQueues->at(clDeviceID)[0].enqueueWriteBuffer(baselines_d, CL_FALSE, 0, baselines.size() * sizeof(outputDataType), reinterpret_cast<void *>(baselines.data()));
        }
    }
    catch (cl::Error &err)
    {
        std::cerr << "OpenCL error H2D transfer: " << std::to_string(err.err()) << "." << std::endl;
        return 1;
    }

    // Generate kernel
    cl::Kernel *kernel;
    std::string *code;
    if (kernelUnderTest == SNR::Kernel::SNR)
    {
        if (ordering == SNR::DataOrdering::DMsSamples)
        {
            code = SNR::getSNRDMsSamplesOpenCL<inputDataType>(conf, inputDataName, observation, observation.getNrSamplesPerBatch(), padding);
        }
        else
        {
            code = SNR::getSNRSamplesDMsOpenCL<inputDataType>(conf, inputDataName, observation, observation.getNrSamplesPerBatch(), padding);
        }
    }
    else if (kernelUnderTest == SNR::Kernel::Max)
    {
        code = SNR::getMaxOpenCL<inputDataType>(conf, ordering, inputDataName, observation, 1, padding);
    }
    else if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
    {
        code = SNR::getMaxStdSigmaCutOpenCL<inputDataType>(conf, ordering, inputDataName, observation, 1, padding, nSigma);
    }
    else if (kernelUnderTest == SNR::Kernel::MedianOfMedians)
    {
        code = SNR::getMedianOfMediansOpenCL<inputDataType>(conf, ordering, inputDataName, observation, 1, medianStep, padding);
    }
    else if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
    {
        code = SNR::getMedianOfMediansAbsoluteDeviationOpenCL<inputDataType>(conf, ordering, inputDataName, observation, 1, medianStep, padding);
    }
    else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
    {
        code = SNR::getAbsoluteDeviationOpenCL<inputDataType>(conf, ordering, inputDataName, observation, 1, padding);
    }
    if (printCode)
    {
        std::cout << *code << std::endl;
    }

    try
    {
        if (kernelUnderTest == SNR::Kernel::SNR)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                kernel = isa::OpenCL::compile("snrDMsSamples" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
            else
            {
                kernel = isa::OpenCL::compile("snrSamplesDMs" + std::to_string(observation.getNrDMs(true) * observation.getNrDMs()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
        }
        else if (kernelUnderTest == SNR::Kernel::Max)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                kernel = isa::OpenCL::compile("max_DMsSamples_" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
        }
        else if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                kernel = isa::OpenCL::compile("maxStdSigmaCut_DMsSamples_" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
        }
        else if (kernelUnderTest == SNR::Kernel::MedianOfMedians)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                kernel = isa::OpenCL::compile("medianOfMedians_DMsSamples_" + std::to_string(medianStep), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
        }
        else if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                kernel = isa::OpenCL::compile("medianOfMediansAbsoluteDeviation_DMsSamples_" + std::to_string(medianStep), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
        }
        else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                kernel = isa::OpenCL::compile("absolute_deviation_DMsSamples_" + std::to_string(observation.getNrSamplesPerBatch()), *code, "-cl-mad-enable -Werror", *clContext, clDevices->at(clDeviceID));
            }
        }
    }
    catch (isa::OpenCL::OpenCLError &err)
    {
        std::cerr << err.what() << std::endl;
        return 1;
    }

    // Run OpenCL kernel and CPU control
    std::vector<isa::utils::Statistics<inputDataType>> control(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs());
    std::vector<outputDataType> medians_control;
    std::vector<outputDataType> absoluteDeviations_control;
    std::vector<outputDataType> stdevs_control;
    if (kernelUnderTest == SNR::Kernel::MedianOfMedians || kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
    {
        medians_control.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType)));
    }
    else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
    {
        absoluteDeviations_control.resize(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType)));
    }
    else if ( kernelUnderTest == SNR::Kernel::MaxStdSigmaCut )
    {
        stdevs_control.resize(observation.getNrSynthesizedBeams() * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType)));
    }
    try
    {
        cl::NDRange global;
        cl::NDRange local;

        if (kernelUnderTest == SNR::Kernel::SNR || kernelUnderTest == SNR::Kernel::Max || kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                global = cl::NDRange(conf.getNrThreadsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
                local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
            }
            else
            {
                global = cl::NDRange((observation.getNrDMs(true) * observation.getNrDMs()) / conf.getNrItemsD0(), observation.getNrSynthesizedBeams());
                local = cl::NDRange(conf.getNrThreadsD0(), 1);
            }
        }
        else if (kernelUnderTest == SNR::Kernel::MedianOfMedians || kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                global = cl::NDRange(conf.getNrThreadsD0() * (observation.getNrSamplesPerBatch() / medianStep), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
                local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
            }
        }
        else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
        {
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                global = cl::NDRange(observation.getNrSamplesPerBatch() / conf.getNrItemsD0(), observation.getNrDMs(true) * observation.getNrDMs(), observation.getNrSynthesizedBeams());
                local = cl::NDRange(conf.getNrThreadsD0(), 1, 1);
            }
        }
        if (kernelUnderTest == SNR::Kernel::SNR)
        {
            kernel->setArg(0, input_d);
            kernel->setArg(1, output_d);
            kernel->setArg(2, outputIndex_d);
        }
        else if(kernelUnderTest == SNR::Kernel::Max)
        {
            kernel->setArg(0, input_d);
            kernel->setArg(1, output_d);
            kernel->setArg(2, outputIndex_d);
        }
        else if(kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            kernel->setArg(0, input_d);
            kernel->setArg(1, output_d);
            kernel->setArg(2, outputIndex_d);
            kernel->setArg(3, stdevs_d);
        }
        else if (kernelUnderTest == SNR::Kernel::MedianOfMedians)
        {
            kernel->setArg(0, input_d);
            kernel->setArg(1, output_d);
        }
        else if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
        {
            kernel->setArg(0, baselines_d);
            kernel->setArg(1, input_d);
            kernel->setArg(2, output_d);
        }
        else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
        {
            kernel->setArg(0, baselines_d);
            kernel->setArg(1, input_d);
            kernel->setArg(2, output_d);
        }
        clQueues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, 0);
        clQueues->at(clDeviceID)[0].enqueueReadBuffer(output_d, CL_TRUE, 0, output.size() * sizeof(outputDataType), reinterpret_cast<void *>(output.data()));
        if (kernelUnderTest == SNR::Kernel::SNR || kernelUnderTest == SNR::Kernel::Max || kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(outputIndex_d, CL_TRUE, 0, outputIndex.size() * sizeof(unsigned int), reinterpret_cast<void *>(outputIndex.data()));
        }
        if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            clQueues->at(clDeviceID)[0].enqueueReadBuffer(stdevs_d, CL_TRUE, 0, stdevs.size() * sizeof(outputDataType), reinterpret_cast<void *>(stdevs.data()));
        }

    }
    catch (cl::Error &err)
    {
        std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
        return 1;
    }
    if (kernelUnderTest == SNR::Kernel::SNR)
    {
        for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
        {
            for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
            {
                for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                {
                    control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm] = isa::utils::Statistics<inputDataType>();
                }
            }
            if (ordering == SNR::DataOrdering::DMsSamples)
            {
                for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                {
                    for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                    {
                        for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                        {
                            control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].addElement(input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + sample]);
                        }
                    }
                }
            }
            else
            {
                for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                {
                    for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
                    {
                        for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                        {
                            control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].addElement(input[(beam * observation.getNrSamplesPerBatch() * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (sample * observation.getNrDMs(true) * observation.getNrDMs(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs(false, padding / sizeof(inputDataType))) + dm]);
                        }
                    }
                }
            }
        }
    }
    else if (kernelUnderTest == SNR::Kernel::MedianOfMedians)
    {
        SNR::medianOfMedians(medianStep, input, medians_control, observation, padding);
    }
    else if (kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
    {
        SNR::medianOfMediansAbsoluteDeviation(medianStep, baselines, input, medians_control, observation, padding);
    }
    else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
    {
        SNR::absoluteDeviation(baselines, input, absoluteDeviations_control, observation, padding);
    }
    else if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
    {
        SNR::stdSigmaCut(input, stdevs_control, observation, padding, nSigma);
    }

    for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
    {
        for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
        {
            for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
            {
                if (kernelUnderTest == SNR::Kernel::SNR)
                {
                    if (!isa::utils::same(output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm], static_cast<outputDataType>((control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].getMax() - control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].getMean()) / control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].getStandardDeviation()), static_cast<outputDataType>(1e-2)))
                    {
                        wrongSamples++;
                    }
                    if (outputIndex.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) != maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm))
                    {
                        wrongPositions++;
                    }
                }
                else if (kernelUnderTest == SNR::Kernel::Max)
                {
                    if (!isa::utils::same(output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm], static_cast<outputDataType>(input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm)]), static_cast<outputDataType>(1e-2)))
                    {
                        wrongSamples++;
                    }
                    if (outputIndex.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) != maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm))
                    {
                        wrongPositions++;
                    }
                }
                else if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
                {
                    if (!isa::utils::same(output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm], static_cast<outputDataType>(input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm)]), static_cast<outputDataType>(1e-2)))
                    {
                        wrongSamples++;
                    }
                    if (outputIndex.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) != maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm))
                    {
                        wrongPositions++;
                    }
                    if (!isa::utils::same(stdevs[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm], stdevs_control.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm), static_cast<outputDataType>(1e-2)))
                    {
                      wrongSamples_stdev++;
                    }
                }
                else if (kernelUnderTest == SNR::Kernel::MedianOfMedians || kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
                {
                    if (medianStep == observation.getNrSamplesPerBatch())
                    {
                        if (!isa::utils::same(output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm], medians_control[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm], static_cast<outputDataType>(1e-2)))
                        {
                            wrongSamples++;
                        }
                    }
                    else
                    {
                        for (unsigned int step = 0; step < observation.getNrSamplesPerBatch() / medianStep; step++)
                        {
                            if (!isa::utils::same(output[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + step], medians_control[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + step], static_cast<outputDataType>(1e-2)))
                            {
                                wrongSamples++;
                            }
                        }
                    }
                }
                else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
                {
                    for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                    {
                        if (!isa::utils::same(output.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + sample), absoluteDeviations_control.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + sample), static_cast<outputDataType>(1e-03)))
                        {
                            wrongSamples++;
                        }
                    }
                }
            }
        }
    }

    if (printResults)
    {
        for (unsigned int beam = 0; beam < observation.getNrSynthesizedBeams(); beam++)
        {
            std::cout << "Beam: " << beam << std::endl;
            for (unsigned int subbandingDM = 0; subbandingDM < observation.getNrDMs(true); subbandingDM++)
            {
                for (unsigned int dm = 0; dm < observation.getNrDMs(); dm++)
                {
                    if (kernelUnderTest == SNR::Kernel::SNR)
                    {
                        std::cout << output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm] << "," << (control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].getMax() - control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].getMean()) / control[(beam * observation.getNrDMs(true) * observation.getNrDMs()) + (subbandingDM * observation.getNrDMs()) + dm].getStandardDeviation() << " ; ";
                        std::cout << outputIndex.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << "," << maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << "  ";
                    }
                    else if (kernelUnderTest == SNR::Kernel::Max)
                    {
                        std::cout << output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm] << "," << input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm)] << " ; ";
                        std::cout << outputIndex.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << "," << maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << " ";
                    }
                    else if (kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
                    {
                        std::cout << output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(float))) + (subbandingDM * observation.getNrDMs()) + dm] << "," << input[(beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(inputDataType))) + maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm)] << " ; ";
                        std::cout << outputIndex.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << "," << maxSample.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << " ; ";
                        std::cout << stdevs.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << "," << stdevs_control.at((beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(unsigned int))) + (subbandingDM * observation.getNrDMs()) + dm) << " ";
                    }
                    else if (kernelUnderTest == SNR::Kernel::MedianOfMedians || kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
                    {
                        if (medianStep == observation.getNrSamplesPerBatch())
                        {
                            std::cout << output[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm] << "," << medians_control[(beam * isa::utils::pad(observation.getNrDMs(true) * observation.getNrDMs(), padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs()) + dm] << " ";
                        }
                        else
                        {
                            for (unsigned int step = 0; step < observation.getNrSamplesPerBatch() / medianStep; step++)
                            {
                                std::cout << output[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + step] << "," << medians_control[(beam * observation.getNrDMs(true) * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + (dm * isa::utils::pad(observation.getNrSamplesPerBatch() / medianStep, padding / sizeof(outputDataType))) + step] << " ";
                            }
                            std::cout << std::endl;
                        }
                    }
                    else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
                    {
                        for (unsigned int sample = 0; sample < observation.getNrSamplesPerBatch(); sample++)
                        {
                            std::cout << output.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + sample) << "," << absoluteDeviations_control.at((beam * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (subbandingDM * observation.getNrDMs() * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + (dm * observation.getNrSamplesPerBatch(false, padding / sizeof(outputDataType))) + sample) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << std::endl;
    }

    if (wrongSamples > 0)
    {
        if (kernelUnderTest == SNR::Kernel::SNR || kernelUnderTest == SNR::Kernel::Max || kernelUnderTest == SNR::Kernel::MaxStdSigmaCut)
        {
            std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast<uint64_t>(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs()) << "%)." << std::endl;
        }
        else if (kernelUnderTest == SNR::Kernel::MedianOfMedians || kernelUnderTest == SNR::Kernel::MedianOfMediansAbsoluteDeviation)
        {
            std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast<uint64_t>(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * (observation.getNrSamplesPerBatch() / medianStep)) << "%)." << std::endl;
        }
        else if (kernelUnderTest == SNR::Kernel::AbsoluteDeviation)
        {
            std::cout << "Wrong samples: " << wrongSamples << " (" << (wrongSamples * 100.0) / static_cast<uint64_t>(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs() * observation.getNrSamplesPerBatch()) << "%)." << std::endl;
        }
    }
    else if (wrongPositions > 0)
    {
        std::cout << "Wrong positions: " << wrongPositions << " (" << (wrongPositions * 100.0) / static_cast<uint64_t>(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs()) << "%)." << std::endl;
    }
    else if ( wrongSamples_stdev > 0 )
    {
      std::cout << "Wrong StdDev samples: " << wrongSamples_stdev << " (" << (wrongSamples_stdev * 100.0) / static_cast<uint64_t>(observation.getNrSynthesizedBeams() * observation.getNrDMs(true) * observation.getNrDMs()) << "%)." << std::endl;
    }
    else
    {
        std::cout << "TEST PASSED." << std::endl;
    }
    return 0;
}
