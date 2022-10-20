
# Parallel Programming - Weather Analysis (CMP3752M_Assessment_01)


## Overview
A program to analyse historical weather records from Lincolnshire. The provided data set includes records of air temperature collected over a period of more than 80 years from five weather stations in Lincolnshire. The program loads the dataset and calculates statistical summaries of temperature, including minimum, maximum, mean, and standard deviation.

All statistical calculations are performed on parallel hardware and implemented by a parallel software component written in
OpenCL. The program also reports memory transfer, kernal execution, and total program execution times, which were optimised for performance. No existing parallel libraries (e.g. Boost.Compute) were used in the project.

## Requirements

The presented code was developed and tested on Windows 10, Visual Studio 2019, Intel Quad Core CPU, GTX 970 and is based on the Tutorial 3 solution from the in class workshop tutorials

 
## Windows Setup
 - OS + IDE: Windows 10, Visual Studio 2019
 - OpenCL SDK: the SDK enables you to develop and compile the OpenCL code. In our case, we use [Intel SDK for OpenCL Applications](https://software.intel.com/en-us/intel-opencl). You are not tied to that choice, however, and can use SDKs by NVidia or AMD - just remember to make modifications in the project include paths. Each SDK comes with a range of additional tools which make development of OpenCL programs easier.
 - OpenCL runtime: the runtime drivers are necessary to run the OpenCL code on your hardware. Both NVidia and AMD GPUs have OpenCL runtime included with their card drivers. For CPUs, you will need to install a dedicated driver by [Intel](https://software.intel.com/en-us/articles/opencl-drivers) or APP SDK for older AMD processors. It seems that AMD’s OpenCL support for newer CPU models was dropped unfortunately. You can check the existing OpenCL support on your PC using [GPU Caps Viewer](http://www.ozone3d.net/gpu_caps_viewer/).


## Code Execution: 

Download the provoided .zip file to a suitable location for execution. Extract the folder and open the `Solution1.sln` file with VS2019. Once open, select `main.cpp` and either use the local windows debugger, or build the project (Ctrl+B) and execute in a seperate command line window, such as: `C:\filepath\CMP3752M_Assessment_01\x64\Debug\Solution1` then execute `Solution1.exe`


## Dataset Information

The provided data files contain records of air temperature collected over a period of more than 80 years from five weather stations in Lincolnshire: Barkston Heath, Scampton, Waddington, Cranwell and Coningsby. The original file is called "temp_lincolnshire.txt" and the short dataset is in "temp_lincolnshire_short.txt".

Each column corresponds to the following category:

   1. Weather station name
   2. Year the record was collected
   3. Month
   4. Day
   5. Time (HHMM)
   6. Air temperature (degrees Celsius)
   
   
## Summary of Implementation
When the code is executed, various parts of the code are initialised, such as gathering the computing device and enabling profiling for the queue. Afterward, the dataset is imported using the input stream class. The extracted temperature values are then first converted from string to float to maintain accuracy, however, are then multiplied by 100 and stored as int. This is done to maintain accuracy to 2dp while still allowing the kernels to operate on integers.

Next, the serial part of the code is ran, to get a baseline comparison for the other algorithms. Subsequently, events are initialised to track timings. The padding for the input data is created with the value of 0. Various other vector sizes and pre - required values are defined here, as well as the buffers created. The min, max, and sum kernels are then called and copied into memory. Simple serial calculations are performed on the sum to find the mean, which can be done here as it doesn’t affect execution time. This is then repeated after the mean has been found for the variance. The variance kernel includes the ‘input_size’ integer, which is used to ignore the additional padded values created. Finally, the outputted values are displayed to the user alongside the memory transfer + execution times.


### Reference

https://github.com/alanmillard/OpenCL-Tutorials


