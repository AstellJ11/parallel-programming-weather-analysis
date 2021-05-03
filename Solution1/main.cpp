#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Utils.h"

// -------------------------------------------------------- //
// CMP3752M - 2021: Parallel Programming - Assessment 01
// Written by James Astell (17668733)
// -------------------------------------------------------- //

// ----------------------------- Summary of Implementation ----------------------------- //
//
//	To do
//
//

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	// Handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}


	// ----------------------------- MAIN INIT ----------------------------- //

	try {
		// Host operations - Select computing devices (GPU default)
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		// Enable profiling for the queue
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		// Build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << endl;
			throw err;
		}

		// Display device properties
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // Get device
		cout << " " << endl;
		cout << "Global Memory Size: " << device.getInfo <CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
		cout << "Local Memory Size: " << device.getInfo <CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
		cout << " " << endl;
		cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
		cout << "Max Device Dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << endl;
		cout << "Max Work Items Per Workgroup: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << endl;
		cout << " " << endl;

		typedef int mytype;

		// Declare all vector stores
		//vector<string> place = {};
		//vector<int> year = {};
		//vector<int> month = {};
		//vector<int> day = {};
		//vector<int> time = {};
		vector<mytype> temperature = {};


		// ----------------------------- FILE IMPORTING ----------------------------- //

		// Input stream class for import file
		ifstream myfile("temp_lincolnshire_short.txt");
		string line;

		if (myfile.is_open()) {

			// For each line in import file
			while (getline(myfile, line)) {
				vector<string> line_vec = {};
				istringstream each_line(line);

				// For each line in extracted line with space delim
				while (getline(each_line, line, ' ')) {
					line_vec.push_back(line);
				}

				int line_int = stoi(line_vec.back()); // Convert string to int
				temperature.push_back(line_int); // Add final string of each line to back of vector
			}
		}
		else {
			cout << "Error importing the data!" << endl;
		}


		// ----------------------------- SERIAL CODE ----------------------------- //

		// Find the max element 
		int max_temp = *max_element(temperature.begin(), temperature.end());

		// Find the min element 
		int min_temp = *min_element(temperature.begin(), temperature.end());

		// Find the mean value
		auto n = temperature.size();
		float average = 0.0f;
		if (n != 0) {
			average = accumulate(temperature.begin(), temperature.end(), 0.0) / n;
		}

		// Find the standard deviation value
		float var = 0;
		float numPoints = temperature.size();
		for (n = 0; n < numPoints; n++)
		{
			var += (temperature[n] - average) * (temperature[n] - average);
		}

		var /= numPoints;
		float sd = sqrt(var);

		cout << "Serial Outputs:";
		cout << "\nMax = " << max_temp << " Min = " << min_temp << " Mean = " << average << " SD = " << sd << endl;
		cout << " " << endl;


		// ----------------------------- SERIAL CODE ENDS ----------------------------- //

		// Create events for tracking execution time
		cl::Event prof_event;
		cl::Event A_event;
		cl::Event B_event;

		// Memory allocation
		// Host - input
		size_t local_size = 256;
		size_t padding_size = temperature.size() % local_size;

		// Data padding
		if (padding_size) {
			// Create an extra vector with neutral values
			vector<int> A_ext(local_size - padding_size, 0);
			// Append that extra vector to our input
			temperature.insert(temperature.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = temperature.size();  // Number of input elements
		size_t input_size = temperature.size() * sizeof(mytype);  // Size in bytes
		size_t nr_groups = input_elements / local_size;

		// Host - output
		size_t output_size = input_size; // Size in bytes
		//size_t output_size = B.size() * sizeof(mytype);  // Size in bytes

		vector<mytype> min_output(input_elements);
		vector<mytype> max_output(input_elements);

		// Device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);  // Input buffer

		// Output buffers
		cl::Buffer buffer_min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_max(context, CL_MEM_READ_WRITE, output_size);

		// Device operations
		// Copy array A to and initialise other arrays on device memory
		// A is the input vector
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temperature[0]);

		// Zero other buffers on device memory
		queue.enqueueFillBuffer(buffer_min, 0, 0, output_size); 
		queue.enqueueFillBuffer(buffer_max, 0, 0, output_size);

		// ----------------------------- Execute min kernel ----------------------------- //

		cl::Kernel min_reduce = cl::Kernel(program, "min_reduce");
		min_reduce.setArg(0, buffer_A);
		min_reduce.setArg(1, buffer_min);
		min_reduce.setArg(2, cl::Local(local_size * sizeof(mytype)));

		// ----------------------------- Execute max kernel ----------------------------- //

		cl::Kernel max_reduce = cl::Kernel(program, "max_reduce");
		max_reduce.setArg(0, buffer_A);
		max_reduce.setArg(1, buffer_max);
		max_reduce.setArg(2, cl::Local(local_size * sizeof(mytype)));


		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(min_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));
		queue.enqueueNDRangeKernel(max_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));


		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0, output_size, &min_output[0]);
		queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0, output_size, &max_output[0]);

		// To check vectors
		//cout << "A = " << temperature << endl;
		//cout << "B = " << B << endl;

		cout << "Parallel Outputs: " << endl;
		cout << "Min = " << min_output[0] << endl;
		cout << "Max = " << max_output[0] << endl;


		// Display the kernel + upload/download execution time
		//cout << "\nKernel execution time for seperate kernels [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		//cout << "\nUpload time for vector A [ns]: " << A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		//cout << "Upload time for vector B [ns]: " << B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		//// Display profiling information
		//cout << "\nProfiling information: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
};