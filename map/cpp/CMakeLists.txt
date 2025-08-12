cmake_minimum_required(VERSION 3.12)
project(SIDIS_Computation LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find required packages
find_package(PkgConfig REQUIRED)

# Find LHAPDF
find_program(LHAPDF_CONFIG lhapdf-config)
if(LHAPDF_CONFIG)
    execute_process(COMMAND ${LHAPDF_CONFIG} --cppflags OUTPUT_VARIABLE LHAPDF_CPPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${LHAPDF_CONFIG} --ldflags OUTPUT_VARIABLE LHAPDF_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(LHAPDF_CPPFLAGS)
    separate_arguments(LHAPDF_LDFLAGS)
    message(STATUS "Found LHAPDF: ${LHAPDF_CONFIG}")
else()
    message(FATAL_ERROR "lhapdf-config not found. Please install LHAPDF.")
endif()

# Find APFEL++
find_program(APFELXX_CONFIG apfelxx-config)
if(APFELXX_CONFIG)
    execute_process(COMMAND ${APFELXX_CONFIG} --cppflags OUTPUT_VARIABLE APFELXX_CPPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${APFELXX_CONFIG} --ldflags OUTPUT_VARIABLE APFELXX_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(APFELXX_CPPFLAGS)
    separate_arguments(APFELXX_LDFLAGS)
    message(STATUS "Found APFEL++: ${APFELXX_CONFIG}")
else()
    message(FATAL_ERROR "apfelxx-config not found. Please install APFEL++.")
endif()

# Find NangaParbat
find_program(NANGAPARBAT_CONFIG NangaParbat-config)
if(NANGAPARBAT_CONFIG)
    execute_process(COMMAND ${NANGAPARBAT_CONFIG} --cppflags OUTPUT_VARIABLE NANGAPARBAT_CPPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${NANGAPARBAT_CONFIG} --ldflags OUTPUT_VARIABLE NANGAPARBAT_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
    separate_arguments(NANGAPARBAT_CPPFLAGS)
    separate_arguments(NANGAPARBAT_LDFLAGS)
    message(STATUS "Found NangaParbat: ${NANGAPARBAT_CONFIG}")
else()
    message(FATAL_ERROR "NangaParbat-config not found. Please install NangaParbat.")
endif()

# Find yaml-cpp
pkg_check_modules(YAMLCPP REQUIRED yaml-cpp)

# Handle yaml-cpp linking properly
if(YAMLCPP_FOUND)
    message(STATUS "Found yaml-cpp: ${YAMLCPP_VERSION}")
    set(YAMLCPP_LINK_FLAGS ${YAMLCPP_LDFLAGS})
    if(NOT YAMLCPP_LINK_FLAGS)
        # Fallback: try to find library directly
        find_library(YAMLCPP_LIBRARY yaml-cpp 
            PATHS ${YAMLCPP_LIBRARY_DIRS} /opt/homebrew/lib /usr/local/lib)
        if(YAMLCPP_LIBRARY)
            set(YAMLCPP_LINK_FLAGS ${YAMLCPP_LIBRARY})
        endif()
    endif()
else()
    message(FATAL_ERROR "yaml-cpp not found. Please install yaml-cpp.")
endif()

# Find GSL
find_package(PkgConfig REQUIRED)
pkg_check_modules(GSL gsl)

if(GSL_FOUND)
    message(STATUS "Found GSL via pkg-config: ${GSL_LIBRARIES}")
    # Use the full linking flags from pkg-config
    set(GSL_LINK_LIBRARIES ${GSL_LDFLAGS})
else()
    # Try to find GSL using find_package as fallback
    find_package(GSL)
    if(GSL_FOUND)
        set(GSL_INCLUDE_DIRS ${GSL_INCLUDE_DIR})
        set(GSL_LINK_LIBRARIES ${GSL_LIBRARIES})
        message(STATUS "Found GSL via find_package: ${GSL_LIBRARIES}")
    else()
        # Check for Homebrew GSL installation (macOS)
        if(APPLE)
            find_path(GSL_INCLUDE_DIR gsl/gsl_math.h
                PATHS /opt/homebrew/include /usr/local/include
                NO_DEFAULT_PATH)
            find_library(GSL_LIBRARY gsl
                PATHS /opt/homebrew/lib /usr/local/lib
                NO_DEFAULT_PATH)
            find_library(GSL_CBLAS_LIBRARY gslcblas
                PATHS /opt/homebrew/lib /usr/local/lib
                NO_DEFAULT_PATH)
            
            if(GSL_INCLUDE_DIR AND GSL_LIBRARY AND GSL_CBLAS_LIBRARY)
                set(GSL_INCLUDE_DIRS ${GSL_INCLUDE_DIR})
                set(GSL_LINK_LIBRARIES ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY} m)
                set(GSL_FOUND TRUE)
                message(STATUS "Found GSL (Homebrew): ${GSL_LINK_LIBRARIES}")
            endif()
        endif()
    endif()
endif()

if(NOT GSL_FOUND)
    message(FATAL_ERROR "GSL not found. Please install GSL.")
endif()

# Find all .cc source files in the current directory (excluding subdirectories)
file(GLOB CC_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# Function to create executable from source file
function(create_executable_from_source source_file)
    # Get the filename without extension for the executable name
    get_filename_component(exec_name ${source_file} NAME_WE)
    
    # Create executable
    add_executable(${exec_name} ${source_file})
    
    # Set compile options
    target_compile_options(${exec_name} PRIVATE
        ${LHAPDF_CPPFLAGS}
        ${APFELXX_CPPFLAGS}
        ${NANGAPARBAT_CPPFLAGS}
        ${YAMLCPP_CFLAGS}
    )
    
    # Set include directories
    target_include_directories(${exec_name} PRIVATE
        ${GSL_INCLUDE_DIRS}
    )
    
    # Link libraries
    target_link_libraries(${exec_name} PRIVATE
        ${LHAPDF_LDFLAGS}
        ${APFELXX_LDFLAGS}
        ${NANGAPARBAT_LDFLAGS}
        ${YAMLCPP_LINK_FLAGS}
        ${GSL_LINK_LIBRARIES}
    )
    
    # Add to the list of targets for installation
    list(APPEND ALL_TARGETS ${exec_name})
    set(ALL_TARGETS ${ALL_TARGETS} PARENT_SCOPE)
endfunction()

# Create executables for all found .cc files
set(ALL_TARGETS "")
foreach(source_file ${CC_SOURCES})
    get_filename_component(filename ${source_file} NAME)
    message(STATUS "Creating executable for: ${filename}")
    create_executable_from_source(${source_file})
endforeach()

# Store targets for installation
set(ALL_EXECUTABLE_TARGETS ${ALL_TARGETS})

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  LHAPDF flags: ${LHAPDF_CPPFLAGS}")
message(STATUS "  APFEL++ flags: ${APFELXX_CPPFLAGS}")
message(STATUS "  NangaParbat flags: ${NANGAPARBAT_CPPFLAGS}")
message(STATUS "  YAML-CPP flags: ${YAMLCPP_CFLAGS}")
message(STATUS "  GSL include: ${GSL_INCLUDE_DIRS}")
message(STATUS "  GSL libraries: ${GSL_LINK_LIBRARIES}")
message(STATUS "")

# Installation rules
if(ALL_EXECUTABLE_TARGETS)
    install(TARGETS ${ALL_EXECUTABLE_TARGETS}
        DESTINATION bin
        COMPONENT runtime)
    message(STATUS "Executables to be installed: ${ALL_EXECUTABLE_TARGETS}")
else()
    message(STATUS "No executables found for installation")
endif()

