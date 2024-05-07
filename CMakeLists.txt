cmake_minimum_required (VERSION 3.26)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("ProbabilitiesLab5")

add_executable (ProbabilitiesLab5 "ProbabilitiesLab5.cpp" )

# Import Boost
find_package(Boost REQUIRED)
target_link_libraries(ProbabilitiesLab5 PRIVATE Boost::boost)

# Import nlohmann/json
include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)
target_link_libraries(ProbabilitiesLab5 PRIVATE nlohmann_json::nlohmann_json)

# Copy samples to bin directory
file(GLOB samples "samples/*")
foreach(sample ${samples})
	get_filename_component(sampleName ${sample} NAME)
	configure_file(${sample} "samples/${sampleName}" COPYONLY)
endforeach()