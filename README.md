
# CMP3752M_Assessment_01

## Requirements

The presented code was developed and tested on Windows 10, Visual Studio 2019, Intel Quad Core CPU, GTX 970 and is based on the 'tutorial 3' solution from the in class workshop tutorials

 
## Windows Setup
 - OS + IDE: Windows 10, Visual Studio 2019
 - OpenCL SDK: the SDK enables you to develop and compile the OpenCL code. In our case, we use [Intel SDK for OpenCL Applications](https://software.intel.com/en-us/intel-opencl). You are not tied to that choice, however, and can use SDKs by NVidia or AMD - just remember to make modifications in the project include paths. Each SDK comes with a range of additional tools which make development of OpenCL programs easier.
 - OpenCL runtime: the runtime drivers are necessary to run the OpenCL code on your hardware. Both NVidia and AMD GPUs have OpenCL runtime included with their card drivers. For CPUs, you will need to install a dedicated driver by [Intel](https://software.intel.com/en-us/articles/opencl-drivers) or APP SDK for older AMD processors. It seems that AMD’s OpenCL support for newer CPU models was dropped unfortunately. You can check the existing OpenCL support on your PC using [GPU Caps Viewer](http://www.ozone3d.net/gpu_caps_viewer/).


## Code Execution: 

Download the provoided .zip file to a suitable location for execution. Extract the folder and open the 'Solution1.sln' file with VS2019. Once open, select 'main.cpp' and either use the local windows debugger, or build the project (Ctrl+B) and execute in a seperate command line window, such as: `C:\filepath\CMP3752M_Assessment_01\x64\Debug\Solution1` then execute `Solution1.exe`


## Dataset Information

The provided data files contain records of air temperature collected over a period of more than 80 years from five weather stations in Lincolnshire: Barkston Heath, Scampton, Waddington, Cranwell and Coningsby. The original file is called "temp_lincolnshire.txt" and the short dataset is in "temp_lincolnshire_short.txt".

Each column corresponds to the following category:

   1. Weather station name
   2. Year the record was collected
   3. Month
   4. Day
   5. Time (HHMM)
   6. Air temperature (degrees Celsius)

### Reference

https://github.com/alanmillard/OpenCL-Tutorials/blob/master/README.md


