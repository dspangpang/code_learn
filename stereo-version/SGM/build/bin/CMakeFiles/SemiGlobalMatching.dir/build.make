# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jzz/code_learn/stereo-version/SGM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jzz/code_learn/stereo-version/SGM/build

# Include any dependencies generated for this target.
include bin/CMakeFiles/SemiGlobalMatching.dir/depend.make

# Include the progress variables for this target.
include bin/CMakeFiles/SemiGlobalMatching.dir/progress.make

# Include the compile flags for this target's objects.
include bin/CMakeFiles/SemiGlobalMatching.dir/flags.make

bin/CMakeFiles/SemiGlobalMatching.dir/main.cpp.o: bin/CMakeFiles/SemiGlobalMatching.dir/flags.make
bin/CMakeFiles/SemiGlobalMatching.dir/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jzz/code_learn/stereo-version/SGM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object bin/CMakeFiles/SemiGlobalMatching.dir/main.cpp.o"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SemiGlobalMatching.dir/main.cpp.o -c /home/jzz/code_learn/stereo-version/SGM/src/main.cpp

bin/CMakeFiles/SemiGlobalMatching.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiGlobalMatching.dir/main.cpp.i"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jzz/code_learn/stereo-version/SGM/src/main.cpp > CMakeFiles/SemiGlobalMatching.dir/main.cpp.i

bin/CMakeFiles/SemiGlobalMatching.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiGlobalMatching.dir/main.cpp.s"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jzz/code_learn/stereo-version/SGM/src/main.cpp -o CMakeFiles/SemiGlobalMatching.dir/main.cpp.s

bin/CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.o: bin/CMakeFiles/SemiGlobalMatching.dir/flags.make
bin/CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.o: ../src/SemiGlobalMatching.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jzz/code_learn/stereo-version/SGM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object bin/CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.o"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.o -c /home/jzz/code_learn/stereo-version/SGM/src/SemiGlobalMatching.cpp

bin/CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.i"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jzz/code_learn/stereo-version/SGM/src/SemiGlobalMatching.cpp > CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.i

bin/CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.s"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jzz/code_learn/stereo-version/SGM/src/SemiGlobalMatching.cpp -o CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.s

bin/CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.o: bin/CMakeFiles/SemiGlobalMatching.dir/flags.make
bin/CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.o: ../src/sgm_util.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jzz/code_learn/stereo-version/SGM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object bin/CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.o"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.o -c /home/jzz/code_learn/stereo-version/SGM/src/sgm_util.cpp

bin/CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.i"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jzz/code_learn/stereo-version/SGM/src/sgm_util.cpp > CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.i

bin/CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.s"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jzz/code_learn/stereo-version/SGM/src/sgm_util.cpp -o CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.s

bin/CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.o: bin/CMakeFiles/SemiGlobalMatching.dir/flags.make
bin/CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.o: ../src/stdafx.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jzz/code_learn/stereo-version/SGM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object bin/CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.o"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.o -c /home/jzz/code_learn/stereo-version/SGM/src/stdafx.cpp

bin/CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.i"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jzz/code_learn/stereo-version/SGM/src/stdafx.cpp > CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.i

bin/CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.s"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jzz/code_learn/stereo-version/SGM/src/stdafx.cpp -o CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.s

# Object files for target SemiGlobalMatching
SemiGlobalMatching_OBJECTS = \
"CMakeFiles/SemiGlobalMatching.dir/main.cpp.o" \
"CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.o" \
"CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.o" \
"CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.o"

# External object files for target SemiGlobalMatching
SemiGlobalMatching_EXTERNAL_OBJECTS =

bin/SemiGlobalMatching: bin/CMakeFiles/SemiGlobalMatching.dir/main.cpp.o
bin/SemiGlobalMatching: bin/CMakeFiles/SemiGlobalMatching.dir/SemiGlobalMatching.cpp.o
bin/SemiGlobalMatching: bin/CMakeFiles/SemiGlobalMatching.dir/sgm_util.cpp.o
bin/SemiGlobalMatching: bin/CMakeFiles/SemiGlobalMatching.dir/stdafx.cpp.o
bin/SemiGlobalMatching: bin/CMakeFiles/SemiGlobalMatching.dir/build.make
bin/SemiGlobalMatching: /usr/local/lib/libopencv_gapi.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_highgui.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_ml.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_objdetect.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_photo.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_stitching.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_video.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_videoio.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_imgcodecs.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_dnn.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_calib3d.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_features2d.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_flann.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_imgproc.so.4.5.4
bin/SemiGlobalMatching: /usr/local/lib/libopencv_core.so.4.5.4
bin/SemiGlobalMatching: bin/CMakeFiles/SemiGlobalMatching.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jzz/code_learn/stereo-version/SGM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable SemiGlobalMatching"
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SemiGlobalMatching.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bin/CMakeFiles/SemiGlobalMatching.dir/build: bin/SemiGlobalMatching

.PHONY : bin/CMakeFiles/SemiGlobalMatching.dir/build

bin/CMakeFiles/SemiGlobalMatching.dir/clean:
	cd /home/jzz/code_learn/stereo-version/SGM/build/bin && $(CMAKE_COMMAND) -P CMakeFiles/SemiGlobalMatching.dir/cmake_clean.cmake
.PHONY : bin/CMakeFiles/SemiGlobalMatching.dir/clean

bin/CMakeFiles/SemiGlobalMatching.dir/depend:
	cd /home/jzz/code_learn/stereo-version/SGM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jzz/code_learn/stereo-version/SGM /home/jzz/code_learn/stereo-version/SGM/src /home/jzz/code_learn/stereo-version/SGM/build /home/jzz/code_learn/stereo-version/SGM/build/bin /home/jzz/code_learn/stereo-version/SGM/build/bin/CMakeFiles/SemiGlobalMatching.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : bin/CMakeFiles/SemiGlobalMatching.dir/depend

