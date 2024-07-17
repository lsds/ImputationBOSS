cmake_minimum_required(VERSION 3.10)
project(BulkEngine)

# Configure CCache if available
find_program(CCACHE_EXECUTABLE ccache)
mark_as_advanced(CCACHE_EXECUTABLE)
if(CCACHE_EXECUTABLE)
  foreach(LANG C CXX)
    if(NOT DEFINED CMAKE_${LANG}_COMPILER_LAUNCHER AND NOT CMAKE_${LANG}_COMPILER MATCHES ".*/ccache")
      message(STATUS "Enabling ccache for ${LANG}")
      set(CMAKE_${LANG}_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE} CACHE STRING "")
    endif()
  endforeach()
endif()

############################## Custom build modes ###############################

set(CMAKE_CXX_FLAGS_SANITIZE "-fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
  CACHE STRING
  "Flags used by the C++ compiler during Sanitize builds."
  FORCE)
set(CMAKE_C_FLAGS_SANITIZE "-fno-omit-frame-pointer -fsanitize=address,signed-integer-overflow,null,alignment,bounds,function,return,vla-bound -O0 -g"
  CACHE STRING
  "Flags used by the C compiler during Sanitize builds."
  FORCE)
set(CMAKE_EXE_LINKER_FLAGS_SANITIZE
  ${CMAKE_EXE_LINKER_FLAGS_DEBUG} CACHE STRING
  "Flags used for linking binaries during Sanitize builds."
  FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} CACHE STRING
  "Flags used by the shared libraries linker during Sanitize builds."
  FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_SANITIZE		  CMAKE_EXE_LINKER_FLAGS_SANITIZE
  CMAKE_C_FLAGS_SANITIZE		  CMAKE_SHARED_LINKER_FLAGS_SANITIZE
  )

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Sanitize."
  FORCE)

############################### External Projects ###############################

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE) # make sure all depedent libraries are compiled with that option

include(ExternalProject)

if(WIN32)
  set(XGBOOST_PATCH_COMMAND git checkout include && git checkout src && python -m patch ${BulkEngine_SOURCE_DIR}/xgboost.patch)
else()
  set(XGBOOST_PATCH_COMMAND git checkout include && git checkout src && patch --binary -l -p1 -i ${BulkEngine_SOURCE_DIR}/xgboost.patch)
endif(WIN32)

ExternalProject_Add(xgboost
    GIT_REPOSITORY "https://github.com/dmlc/xgboost.git"
    GIT_TAG v1.6.2
    #GIT_SUBMODULES_RECURSE  ON
    DOWNLOAD_DIR $ENV{HOME}/.cmake-downloads
    CMAKE_ARGS -DBUILD_STATIC_LIB=ON -DBUILD_SHARED_LIBS=OFF -DFORCE_SHARED_CRT=ON
               -DCMAKE_INSTALL_PREFIX=${BulkEngine_BINARY_DIR}/deps -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${DCMAKE_CXX_FLAGS}
    BUILD_BYPRODUCTS ${BulkEngine_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}xgboost${CMAKE_STATIC_LIBRARY_SUFFIX}
                     ${BulkEngine_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}dmlc${CMAKE_STATIC_LIBRARY_SUFFIX}
    PATCH_COMMAND ${XGBOOST_PATCH_COMMAND}
    )

# needed for xgboost (because of static linking)
find_package(OpenMP REQUIRED)

ExternalProject_Add(robin-hood
  URL https://github.com/martinus/robin-hood-hashing/archive/refs/tags/3.11.5.zip
  DOWNLOAD_DIR $ENV{HOME}/.cmake-downloads/${CMAKE_PROJECT_NAME}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/src/include
                  ${BulkEngine_BINARY_DIR}/deps/include
)

####################### Other Options and Library Dependencies #######################

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(ENABLE_AVX2 ON CACHE BOOL "")
set(ENABLE_AVX512 OFF CACHE BOOL "")

#################################### Bulk Engine Plugin ####################################

if(MSVC)
  # not making any difference on Windows
  # and helps Visual Studio to parse it correctly as a target
  set(LibraryType SHARED)
else()
  set(LibraryType MODULE)
endif(MSVC)

add_library(BulkEngine ${LibraryType} "Source/Bulk.cpp" "Source/BulkExpression.cpp" "Source/BulkProperties.cpp" "Source/Serialization/TableDataLoader.cpp")
add_dependencies(BulkEngine Core)
target_compile_options(BulkEngine PUBLIC -DARROW_STATIC)
if(WIN32)
    target_compile_options(BulkEngine PUBLIC /bigobj)
endif(WIN32)

target_link_libraries(BulkEngine Threads::Threads)
set_target_properties(BulkEngine PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(BulkEngine PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
if(MSVC)
    target_compile_options(BulkEngine PUBLIC "/Zc:__cplusplus")
    target_compile_options(BulkEngine PUBLIC "/EHsc")
    # to link against all needed symbols for loading BulkEngine as a plugin
    target_link_libraries(BulkEngine ${Core_BINARY_DIR}/Core.lib)
endif()
target_include_directories(BulkEngine SYSTEM PUBLIC ${BulkEngine_BINARY_DIR}/deps/include)
target_include_directories(BulkEngine PUBLIC ${Core_SOURCE_DIR}/Source)
target_include_directories(BulkEngine SYSTEM PUBLIC ${Core_BINARY_DIR}/deps/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(BulkEngine rt)
endif()

# xgboost
add_dependencies(BulkEngine xgboost)
target_link_libraries(BulkEngine ${BulkEngine_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}xgboost${CMAKE_STATIC_LIBRARY_SUFFIX})    
target_link_libraries(BulkEngine ${BulkEngine_BINARY_DIR}/deps/lib/${CMAKE_SHARED_LIBRARY_PREFIX}dmlc${CMAKE_STATIC_LIBRARY_SUFFIX})   
target_link_libraries(BulkEngine OpenMP::OpenMP_CXX) # required because of static xgboost

# hashmap/hashing dependencies
target_compile_options(BulkEngine PUBLIC -DUSE_ROBIN_HASHMAP -DUSE_ROBIN_HASHING) #-DUSE_ROBIN_SET)
add_dependencies(BulkEngine robin-hood)    

if(ENABLE_AVX2)
    if(WIN32)
        target_compile_options(BulkEngine PRIVATE "/arch:AVX2")
    else()
        target_compile_options(BulkEngine PRIVATE "-mavx2")
    endif(WIN32)
endif(ENABLE_AVX2)

if(ENABLE_AVX512)
    if(WIN32)
        target_compile_options(BulkEngine PRIVATE "/arch:AVX512")
    else()
        target_compile_options(BulkEngine PRIVATE "-mavx512f")
    endif(WIN32)
endif(ENABLE_AVX512)

#################################### Install ###################################

set_target_properties(BulkEngine PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS BulkEngine LIBRARY DESTINATION lib)

