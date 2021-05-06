#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "Utils.h"

// -------------------------------------------------------- //
// CMP3752M - 2021: Parallel Programming - Assessment 01
// Written by James Astell (17668733)

// ----------------------------- Summary of Implementation ----------------------------- //
//
//	To do
//
//

// ----------------------------- References ----------------------------- //
// In class examples used as the basis for sections of code + kernels
// Serial calculation to find standard deviation:
// https://stackoverflow.com/questions/33268513/calculating-standard-deviation-variance-in-c
// Finding execution time of block of code:
// https://stackoverflow.com/questions/22387586/measuring-execution-time-of-a-function-in-c

// User display help
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
		// Init serial execution time tracker
		using std::chrono::high_resolution_clock;
		using std::chrono::duration_cast;
		using std::chrono::duration;
		using std::chrono::nanoseconds;
		using std::chrono::milliseconds;
		using std::chrono::seconds;

		auto t1 = high_resolution_clock::now(); // Start execution timer

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

		// REMOVE BEFORE SUBMIT
		// REMOVE BEFORE SUBMIT
		// REMOVE BEFORE SUBMIT
		// REMOVE BEFORE SUBMIT
		// REMOVE BEFORE SUBMIT
		// Display device properties
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // Get device
		cout << endl;
		cout << "Global Memory Size: " << device.getInfo <CL_DEVICE_GLOBAL_MEM_SIZE>() << endl;
		cout << "Local Memory Size: " << device.getInfo <CL_DEVICE_LOCAL_MEM_SIZE>() << endl;
		cout << endl;
		cout << "Max Work Group Size: " << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
		cout << "Max Device Dimensions: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << endl;
		cout << "Max Work Items Per Workgroup: " << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>() << endl;
		cout << endl;

		typedef int mytype;  // Create additional name for datatype int


		// ----------------------------- FILE IMPORTING ----------------------------- //

		auto fi_t1 = high_resolution_clock::now(); // Start execution timer

		// Declare vector store
		vector<int> temperature = {};
		vector<int> temperature_int = {};

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

				float line_f = stof(line_vec.back()); // Convert string to float to maintain accuracy
				int test = line_f * 100;  // Multiply float values by 100 and parse to int, as ints can be accepted by kernel while keeping accuracy upto 2dp
				temperature.push_back(test); // Add final string of each line to back of vector

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
		cout << endl;


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

		// Display outputs
		cout << "Serial Outputs:" << endl;
		cout << "Max = " << (max_temp/100) << " Min = " << (min_temp/100) << " Mean = " << (average/100) << " SD = " << (sd/100) << endl;
		cout << endl;

		auto s_t2 = high_resolution_clock::now();  // Stop execution timer

		// Calculate + display execution time
		auto s_ms_int = duration_cast<nanoseconds>(s_t2 - s_t1);  // INT
		duration<double, nano> s_ms_double = s_t2 - s_t1;  // DOUBLE
		cout << "Serial code execution time [ns / s]: " << s_ms_int.count() << " / " << (s_ms_double.count() / 1000000000) << endl;
		cout << endl;


		// ----------------------------- SERIAL CODE ENDS ----------------------------- //

		// Create events for tracking execution time
		// Upload time (enqueueWriteBuffer)
		cl::Event up_event;

		// Execution time (enqueueNDRangeKernel)
		cl::Event min_event;
		cl::Event max_event;
		cl::Event sum_event;
		cl::Event var_event;
		cl::Event var_sum_event;
		cl::Event sort_event;

		// Download time (enqueueReadBuffer)
		cl::Event down_min_event;
		cl::Event down_max_event;
		cl::Event down_sum_event;
		cl::Event down_var_event;
		cl::Event down_var_sum_event;
		cl::Event down_sort_event;

		// Memory allocation
		// Host - input
		vector<int> temperature_no_pad = temperature;

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

		size_t padded_elements = local_size - padding_size;  // Need to ignore padding value when doing calculations such as the mean

		// Host - output
		size_t output_size = input_size; // Size in bytes
		//size_t output_size = B.size() * sizeof(mytype);  // Size in bytes

		vector<mytype> min_output(input_elements);
		vector<mytype> max_output(input_elements);
		vector<mytype> sum_output(input_elements);
		vector<mytype> var_output(input_elements);
		vector<mytype> var_sum_output(input_elements);
		vector<mytype> sort_output(input_elements);

		// Device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);  // Input buffer

		// Output buffers
		cl::Buffer buffer_min(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_max(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_sum(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_var(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_var_sum(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_sort(context, CL_MEM_READ_WRITE, output_size);

		// Device operations
		// Copy array A to and initialise other arrays on device memory
		// A is the input vector
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temperature[0], NULL, &up_event);

		// Zero other buffers on device memory
		queue.enqueueFillBuffer(buffer_min, 0, 0, output_size); 
		queue.enqueueFillBuffer(buffer_max, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_sum, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_var, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_var_sum, 0, 0, output_size);
		queue.enqueueFillBuffer(buffer_sort, 0, 0, output_size);

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

		// ----------------------------- Execute sum kernel ----------------------------- //

		cl::Kernel sum_reduce = cl::Kernel(program, "sum_reduce");
		sum_reduce.setArg(0, buffer_A);
		sum_reduce.setArg(1, buffer_sum);
		sum_reduce.setArg(2, cl::Local(local_size * sizeof(mytype)));


		// ----------------------------- Call + Copy kernels ----------------------------- //

		// Call all kernels in a sequence
		queue.enqueueNDRangeKernel(min_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &min_event);
		queue.enqueueNDRangeKernel(max_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &max_event);
		queue.enqueueNDRangeKernel(sum_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sum_event);

		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_min, CL_TRUE, 0, output_size, &min_output[0], NULL, &down_min_event);
		queue.enqueueReadBuffer(buffer_max, CL_TRUE, 0, output_size, &max_output[0], NULL, &down_max_event);
		queue.enqueueReadBuffer(buffer_sum, CL_TRUE, 0, output_size, &sum_output[0], NULL, &down_sum_event);


		// ----------------------------- Post parallel mean calculation ----------------------------- //

		float f_sum_output = (sum_output[0]/100);  // To prevent rounding
		// Mean = (sum / total)
		float mean_output = (f_sum_output / (input_elements - padded_elements));  // Need to minus added padded values as not true value
		int int_mean_output = (mean_output * 100);  // Output to be used in later kernel as float mean not accepted


		// ----------------------------- Execute variance kernel ----------------------------- //

		// Variance calculation needs to be performed after mean is calculated as mean is used in kernel
		int int_input_size = temperature.size() - padded_elements;  // input_size needs to be an int to be accepted by kernel

		cl::Kernel variance_reduce = cl::Kernel(program, "variance_reduce");
		variance_reduce.setArg(0, buffer_A);
		variance_reduce.setArg(1, buffer_var);
		variance_reduce.setArg(2, cl::Local(local_size * sizeof(mytype)));
		variance_reduce.setArg(3, int_mean_output);
		variance_reduce.setArg(4, int_input_size);

		// Call kernel in a sequence
		queue.enqueueNDRangeKernel(variance_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &var_event);
		queue.enqueueReadBuffer(buffer_var, CL_TRUE, 0, output_size, &var_output[0], NULL, &down_var_event);  // Copy the result from device to host

		// ----------------------------- Execute sum kernel for variance ----------------------------- //

		cl::Kernel variance_sum_reduce = cl::Kernel(program, "variance_sum_reduce");
		variance_sum_reduce.setArg(0, buffer_var);
		variance_sum_reduce.setArg(1, buffer_var_sum);
		variance_sum_reduce.setArg(2, cl::Local(local_size * sizeof(mytype)));

		// Call kernel in a sequence
		queue.enqueueNDRangeKernel(variance_sum_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &var_sum_event);
		queue.enqueueReadBuffer(buffer_var_sum, CL_TRUE, 0, output_size, &var_sum_output[0], NULL, &down_var_sum_event);  // Copy the result from device to host


		// ----------------------------- Post parallel SD calculation ----------------------------- //

 		float f_var_output = (var_sum_output[0]);  // To prevent rounding
		// SD = sqrt(var / total)
		float sd_output = sqrt(f_var_output / ((input_elements - padded_elements) - 1));  // Actual final sd result


		// ----------------------------- Execute sort kernel ----------------------------- //

		cl::Kernel sort_reduce = cl::Kernel(program, "sort_reduce");
		sort_reduce.setArg(0, buffer_A);
		sort_reduce.setArg(1, buffer_sort);
		sort_reduce.setArg(2, int_input_size);

		// Call kernel in a sequence
		queue.enqueueNDRangeKernel(sort_reduce, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size), NULL, &sort_event);
		queue.enqueueReadBuffer(buffer_sort, CL_TRUE, 0, output_size, &sort_output[0], NULL, &down_sort_event);  // Copy the result from device to host


		// ----------------------------- Post parallel sort calculations ----------------------------- //




		// ----------------------------- Display results ----------------------------- //

		cout << "Parallel Outputs: " << endl;

		cout << "25th percentile = " << (sort_output[0]) << endl;
		cout << "Median = " << (mean_output) << endl;
		cout << "75th percentile = " << (mean_output) << endl;
		cout << endl;


		// ----------------------------- Display memory transfer (upload time) ----------------------------- //

		// Overall operation time = sum of memory transfers + kernel execution

		// Display upload time for initial temp vector
		cout << "Upload time for inital temp vector [ns]: " << up_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - up_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << GetFullProfilingInfo(up_event, ProfilingResolution::PROF_US) << endl;  // Display profiling information
		cout << endl;


		// ----------------------------- Display results + profiling info for MIN ----------------------------- //

		cout << "-------------------- Minimum Value --------------------" << endl;
		cout << "Result = " << (min_output[0] / 100) << endl;  // Main result output

		// Display the kernel download + execution time
		cout << "Kernel execution time for minimum value [ns]: " << min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Download time for output vector [ns]: " << down_min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << endl;
		cout << "Overall operation time [ns]: " << (up_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - up_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << endl;
		cout << "-------------------------------------------------------" << endl;
		cout << endl;


		// ----------------------------- Display results + profiling info for MIN ----------------------------- //

		//cout << "==================== Maximum Value ====================" << endl;
		cout << "-------------------- Maximum Value --------------------" << endl;
		cout << "Result = " << (max_output[0] / 100) << endl;  // Main result output

		// Display the kernel download + execution time
		cout << "Kernel execution time for maximum value [ns]: " << max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Download time for output vector [ns]: " << down_max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << endl;
		cout << "Overall operation time [ns]: " << (up_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - up_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << endl;
		cout << "-------------------------------------------------------" << endl;
		cout << endl;


		// ----------------------------- Display results + profiling info for SUM (MEAN) ----------------------------- //

		cout << "--------------------- Mean Value ----------------------" << endl;
		cout << "Result = " << (mean_output) << endl;  // Main result output

		// Display the kernel download + execution time
		cout << "Kernel execution time for mean value [ns]: " << sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Download time for output vector [ns]: " << down_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << endl;
		cout << "Overall operation time [ns]: " << (up_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - up_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << endl;
		cout << "-------------------------------------------------------" << endl;
		cout << endl;


		// ----------------------------- Display results + profiling info for VAR + SUM_VAR (SD) ----------------------------- //

		cout << "-------------- Standard Deviation Value ---------------" << endl;
		cout << "Result = " << (sd_output) << endl;  // Main result output

		// Display the kernel download + execution time
		cout << "Kernel execution time for standard deviation value [ns]: " << (var_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - var_event.getProfilingInfo<CL_PROFILING_COMMAND_START>())
			+ (var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << endl;
		cout << "Download time for output vector [ns]: " << (down_var_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_var_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << endl;
		cout << endl;
		cout << "Overall operation time [ns]: " << (up_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - up_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (var_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - var_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_var_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_var_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) << endl;
		cout << "-------------------------------------------------------" << endl;
		cout << endl;


		// ----------------------------- Total program execution time ----------------------------- //

		auto t2 = high_resolution_clock::now();  // Stop execution timer

		// Overall operation time = sum of memory transfers + kernel execution
		int operation_time = (up_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - up_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_min_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_min_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_max_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_max_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (var_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - var_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_var_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_var_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()) + (down_var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - down_var_sum_event.getProfilingInfo<CL_PROFILING_COMMAND_START>());

		cout << "------------------ Execution Times --------------------" << endl;
		cout << "Total memory transfer + kernel execution time [ns / s]: " << operation_time << " / " << (operation_time / 1000000000) << endl;
		cout << endl;

		// Calculate + display execution time
		auto ms_int = duration_cast<nanoseconds>(t2 - t1);  // INT
		duration<double, nano> ms_double = t2 - t1;  // DOUBLE
		cout << "Total program execution time [ns / s]: " << ms_int.count() << " / " << (ms_double.count() / 1000000000) << endl;
		cout << "-------------------------------------------------------" << endl;
		cout << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;

	}
	return 0;

};


/*
auto fi_ms_int = duration_cast<milliseconds>(fi_t2 - fi_t1);
auto fi_ms_int2 = duration_cast<seconds>(fi_t2 - fi_t1);
cout << "File importing execution time [ms]: " << fi_ms_int.count() << endl;  // INT
cout << "File importing execution time [s]: " << fi_ms_int2.count() << endl;  // INT

duration<double, milli> fi_ms_double = fi_t2 - fi_t1;
duration<double> fi_ms_double2 = fi_t2 - fi_t1;
cout << "File importing execution time [ms]: " << fi_ms_double.count() << endl;  // DOUBLE
cout << "File importing execution time [s]: " << fi_ms_double2.count() << endl;  // DOUBLE
cout << endl;
*/