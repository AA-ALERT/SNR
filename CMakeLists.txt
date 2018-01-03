cmake_minimum_required(VERSION 3.10)
project(SNR VERSION 2.1)
include(GNUInstallDirs)

add_library(snr SHARED
  src/SNR.cpp
)
set_target_properties(snr PROPERTIES 
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER "include/SNR.hpp"
)
target_include_directories(snr PRIVATE include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g0 -march=native -mtune=native")

add_executable(SNRTesting
  src/SNRTest.cpp
)
target_include_directories(SNRTesting PRIVATE include)
target_link_libraries(SNRTesting snr isa_utils isa_opencl astrodata OpenCL)

add_executable(SNRTuning
  src/SNRTuning.cpp
)
target_include_directories(SNRTuning PRIVATE include)
target_link_libraries(SNRTuning snr isa_utils isa_opencl astrodata OpenCL)

install(TARGETS snr SNRTesting SNRTuning
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
