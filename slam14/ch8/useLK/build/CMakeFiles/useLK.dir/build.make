# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/ly_dra/Desktop/slam/slam14/ch8/useLK

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ly_dra/Desktop/slam/slam14/ch8/useLK/build

# Include any dependencies generated for this target.
include CMakeFiles/useLK.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/useLK.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/useLK.dir/flags.make

CMakeFiles/useLK.dir/useLK.cpp.o: CMakeFiles/useLK.dir/flags.make
CMakeFiles/useLK.dir/useLK.cpp.o: ../useLK.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ly_dra/Desktop/slam/slam14/ch8/useLK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/useLK.dir/useLK.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/useLK.dir/useLK.cpp.o -c /home/ly_dra/Desktop/slam/slam14/ch8/useLK/useLK.cpp

CMakeFiles/useLK.dir/useLK.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/useLK.dir/useLK.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ly_dra/Desktop/slam/slam14/ch8/useLK/useLK.cpp > CMakeFiles/useLK.dir/useLK.cpp.i

CMakeFiles/useLK.dir/useLK.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/useLK.dir/useLK.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ly_dra/Desktop/slam/slam14/ch8/useLK/useLK.cpp -o CMakeFiles/useLK.dir/useLK.cpp.s

CMakeFiles/useLK.dir/useLK.cpp.o.requires:

.PHONY : CMakeFiles/useLK.dir/useLK.cpp.o.requires

CMakeFiles/useLK.dir/useLK.cpp.o.provides: CMakeFiles/useLK.dir/useLK.cpp.o.requires
	$(MAKE) -f CMakeFiles/useLK.dir/build.make CMakeFiles/useLK.dir/useLK.cpp.o.provides.build
.PHONY : CMakeFiles/useLK.dir/useLK.cpp.o.provides

CMakeFiles/useLK.dir/useLK.cpp.o.provides.build: CMakeFiles/useLK.dir/useLK.cpp.o


# Object files for target useLK
useLK_OBJECTS = \
"CMakeFiles/useLK.dir/useLK.cpp.o"

# External object files for target useLK
useLK_EXTERNAL_OBJECTS =

useLK: CMakeFiles/useLK.dir/useLK.cpp.o
useLK: CMakeFiles/useLK.dir/build.make
useLK: /usr/local/lib/libopencv_shape.so.3.2.0
useLK: /usr/local/lib/libopencv_stitching.so.3.2.0
useLK: /usr/local/lib/libopencv_superres.so.3.2.0
useLK: /usr/local/lib/libopencv_videostab.so.3.2.0
useLK: /usr/local/lib/libopencv_objdetect.so.3.2.0
useLK: /usr/local/lib/libopencv_calib3d.so.3.2.0
useLK: /usr/local/lib/libopencv_features2d.so.3.2.0
useLK: /usr/local/lib/libopencv_flann.so.3.2.0
useLK: /usr/local/lib/libopencv_highgui.so.3.2.0
useLK: /usr/local/lib/libopencv_ml.so.3.2.0
useLK: /usr/local/lib/libopencv_photo.so.3.2.0
useLK: /usr/local/lib/libopencv_video.so.3.2.0
useLK: /usr/local/lib/libopencv_videoio.so.3.2.0
useLK: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
useLK: /usr/local/lib/libopencv_imgproc.so.3.2.0
useLK: /usr/local/lib/libopencv_core.so.3.2.0
useLK: CMakeFiles/useLK.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ly_dra/Desktop/slam/slam14/ch8/useLK/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable useLK"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/useLK.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/useLK.dir/build: useLK

.PHONY : CMakeFiles/useLK.dir/build

CMakeFiles/useLK.dir/requires: CMakeFiles/useLK.dir/useLK.cpp.o.requires

.PHONY : CMakeFiles/useLK.dir/requires

CMakeFiles/useLK.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/useLK.dir/cmake_clean.cmake
.PHONY : CMakeFiles/useLK.dir/clean

CMakeFiles/useLK.dir/depend:
	cd /home/ly_dra/Desktop/slam/slam14/ch8/useLK/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ly_dra/Desktop/slam/slam14/ch8/useLK /home/ly_dra/Desktop/slam/slam14/ch8/useLK /home/ly_dra/Desktop/slam/slam14/ch8/useLK/build /home/ly_dra/Desktop/slam/slam14/ch8/useLK/build /home/ly_dra/Desktop/slam/slam14/ch8/useLK/build/CMakeFiles/useLK.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/useLK.dir/depend

