cmake_minimum_required(VERSION 3.10)
project(WolframEngine)

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

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake/Mathematica/CMake/Mathematica" ${CMAKE_MODULE_PATH})
find_package(Mathematica COMPONENTS WSTP)

include(ExternalProject)

ExternalProject_Add(spdlog
	URL https://github.com/gabime/spdlog/archive/refs/tags/v1.8.5.tar.gz
  DOWNLOAD_DIR $ENV{HOME}/.cmake-downloads/${CMAKE_PROJECT_NAME}
	CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${WolframEngine_BINARY_DIR}/deps -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
  )

#################################### Targets ####################################

set(ImplementationFiles Source/WolframEngine.cpp)

if(MSVC)
  # not making any difference on Windows
  # and helps Visual Studio to parse it correctly as a target
  set(LibraryType SHARED)
else()
  set(LibraryType MODULE)
endif(MSVC)

add_library(WolframEngine ${LibraryType} ${ImplementationFiles})

if(Mathematica_WSTP_FOUND)
  target_include_directories(WolframEngine PUBLIC ${Mathematica_INCLUDE_DIRS})
  target_link_libraries(WolframEngine ${Mathematica_WSTP_LIBRARY})
  target_compile_options(WolframEngine PUBLIC -DARROW_STATIC)
  if(UNIX AND NOT APPLE)
    target_link_libraries(WolframEngine uuid)
  endif()
  if(NOT WIN32)
    target_link_libraries(WolframEngine dl)
  endif(NOT WIN32)
  target_compile_definitions(WolframEngine PUBLIC "${Mathematica_WSTP_DEFINITIONS}" "-DMATHEMATICA_KERNEL_EXECUTABLE=${Mathematica_KERNEL_EXECUTABLE}")
  if(WIN32)
    add_custom_command(TARGET WolframEngine POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${Mathematica_WSTP_RUNTIME_LIBRARY_DIRS}/wstp64i4.dll"  $<TARGET_FILE_DIR:WolframEngine>)
  endif(WIN32)
endif(Mathematica_WSTP_FOUND)
set_property(TARGET WolframEngine PROPERTY CXX_STANDARD 17)
if(MSVC)
  target_compile_options(WolframEngine PUBLIC "/Zc:__cplusplus")
  target_compile_options(WolframEngine PUBLIC "/EHsc")
    # to link against all needed symbols for loading WolframEngine as a plugin
    target_link_libraries(WolframEngine ${Core_BINARY_DIR}/Core.lib)
endif(MSVC)
target_include_directories(WolframEngine SYSTEM PUBLIC ${WolframEngine_BINARY_DIR}/deps/include)
target_include_directories(WolframEngine PUBLIC ${Core_SOURCE_DIR}/Source)
target_include_directories(WolframEngine SYSTEM PUBLIC ${Core_BINARY_DIR}/deps/include)
if(UNIX AND NOT APPLE)
    target_link_libraries(WolframEngine rt)
endif()
add_dependencies(WolframEngine spdlog)
add_dependencies(WolframEngine Core)

set_target_properties(WolframEngine PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS WolframEngine LIBRARY DESTINATION lib)
if(WIN32)
  install(FILES $<TARGET_FILE_DIR:WolframEngine>/wstp64i4.dll DESTINATION bin)
endif(WIN32)
