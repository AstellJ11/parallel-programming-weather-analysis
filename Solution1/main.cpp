#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

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
	cout << "Application usage:" << endl;

	cout << "  -p : select platform " << endl;
	cout << "  -d : select device" << endl;
	cout << "  -l : list all platforms and devices" << endl;
	cout << "  -h : print this message" << endl;
}

int main(int argc, char** argv) {
	// Handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
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

		// Init serial execution time tracker
		using std::chrono::high_resolution_clock;
		using std::chrono::duration_cast;
		using std::chrono::duration;
		using std::chrono::nanoseconds;
		using std::chrono::milliseconds;
		using std::chrono::seconds;

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

		// Declare vector store
		vector<mytype> temperature = {};


		// ----------------------------- FILE IMPORTING ----------------------------- //

		auto fi_t1 = high_resolution_clock::now(); // Start execution timer

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

		auto fi_t2 = high_resolution_clock::now();  // Stop execution timer

		// Calculate + display execution time
		auto fi_ms_int = duration_cast<nanoseconds>(fi_t2 - fi_t1);  // INT
		duration<double, nano> fi_ms_double = fi_t2 - fi_t1;  // DOUBLE
		cout << "File importing execution time [ns / s]: " << fi_ms_int.count() << " / " << (fi_ms_double.count() / 1000000000) << endl;
		cout << " " << endl;


		// ----------------------------- SERIAL CODE ----------------------------- //

		auto s_t1 = high_resolution_clock::now(); // Start execution timer

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

		cout << "Serial Outputs:" << endl;
		cout << "Max = " << max_temp << " Min = " << min_temp << " Mean = " << average << " SD = " << sd << endl;
		cout << " " << endl;

		auto s_t2 = high_resolution_clock::now();  // Stop execution timer

		// Calculate + display execution time
		auto s_ms_int = duration_cast<nanoseconds>(s_t2 - s_t1);  // INT
		duration<double, nano> s_ms_double = s_t2 - s_t1;  // DOUBLE
		cout << "Serial code execution time [ns / s]: " << s_ms_int.count() << " / " << (s_ms_double.count() / 1000000000) << endl;
		cout << " " << endl;


		// ----------------------------- SERIAL CODE ENDS ----------------------------- //

		// Create events for tracking execution time
		cl::Event prof_event;
		cl::Event min_event;
		cl::Event max_event;

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


		// ----------------------------- Call + Copy kernels ----------------------------- //

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(min_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &min_event);
		queue.enqueueNDRangeKernel(max_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &max_event);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0, output_size, &min_output[0]);
		queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0, output_size, &max_output[0]);

		// To check vectors
		//cout << "A = " << temperature << endl;
		//cout << "B = " << B << endl;


		// ----------------------------- Display results ----------------------------- //

		cout << "Parallel Outputs: " << endl;
		cout << "Min = " << min_output[0] << endl;
		cout << "Max = " << max_output[0] << endl;
		cout << " " << endl;


		// ----------------------------- Display profiling info ----------------------------- //

		// Display the kernel + upload/download execution time
		cout << "Kernel execution time for minimum value [ns]: " << min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout <<  GetFullProfilingInfo(min_event, ProfilingResolution::PROF_US) << endl;  // Display profiling information
		cout << " " << endl;

		cout << "Kernel execution time for maximum value [ns]: " << max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout <<  GetFullProfilingInfo(max_event, ProfilingResolution::PROF_US) << endl;
		cout << " " << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
};


// ----------------------------- References ----------------------------- //
// In class examples used as the basis for sections of code
// Finding execution time of block of code:
// https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c


/*
auto fi_ms_int = duration_cast<milliseconds>(fi_t2 - fi_t1);
auto fi_ms_int2 = duration_cast<seconds>(fi_t2 - fi_t1);
cout << "File importing execution time [ms]: " << fi_ms_int.count() << endl;  // INT
cout << "File importing execution time [s]: " << fi_ms_int2.count() << endl;  // INT

duration<double, milli> fi_ms_double = fi_t2 - fi_t1;
duration<double> fi_ms_double2 = fi_t2 - fi_t1;
cout << "File importing execution time [ms]: " << fi_ms_double.count() << endl;  // DOUBLE
cout << "File importing execution time [s]: " << fi_ms_double2.count() << endl;  // DOUBLE
cout << " " << endl;
*/