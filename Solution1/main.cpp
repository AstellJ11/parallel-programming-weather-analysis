#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "Utils.h"

int main(int argc, char** argv) {
	// Handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	try {
		// Host operations - Select computing devices (GPU default)
		cl::Context context = GetContext(platform_id, device_id);

		// Display the selected device
		cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << endl;

		// Enable profiling for the queue
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Create events for tracking execution time
		cl::Event prof_event;
		cl::Event A_event;
		cl::Event B_event;

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
			cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << :endl;
			throw err;
		}


		// ############################# SERIAL CODE #############################

		// Declare all vector stores
		vector<string> place = {};
		vector<int> year = {};
		vector<int> month = {};
		vector<int> day = {};
		vector<int> time = {};
		vector<int> temp = {};

		// Input stream class for import file
		ifstream myfile("temp_lincolnshire_short.txt");

		string line;

		// For each line in import file
		while (getline(myfile, line))
		{
			vector<string> line_vec = {};
			istringstream each_line(line);

			// For each line in extracted line with space delim
			while (getline(each_line, line, ' ')) {
				line_vec.push_back(line);
			}

			int line_int = stoi(line_vec.back()); // Convert string to int
			temp.push_back(line_int); // Add final string of each line to back of vector
		}

		// Find the max element 
		int max_temp = *max_element(temp.begin(), temp.end());

		// Find the min element 
		int min_temp = *min_element(temp.begin(), temp.end());

		// Find the mean value
		auto n = temp.size();
		float average = 0.0f;
		if (n != 0) {
			average = accumulate(temp.begin(), temp.end(), 0.0) / n;
		}

		// Find the standard deviation value
		float var = 0;
		float numPoints = temp.size();
		for (n = 0; n < numPoints; n++)
		{
			var += (temp[n] - average) * (temp[n] - average);
		}

		var /= numPoints;
		float sd = sqrt(var);

		cout << "Max = " << max_temp << " Min = " << min_temp << " Mean = " << average << " SD = " << sd << endl;

		// ############################# SERIAL CODE ENDS #############################


		// Memory allocation

		size_t vector_elements = temp.size();//number of elements
		size_t vector_size = temp.size() * sizeof(int);//size in bytes

		// Result output
		vector<int> C(vector_elements);

		// Device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, vector_size);
		//cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);


		// Device operations - Copy arrays A and B to device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, vector_size, &temp[0], NULL, &A_event);
		//queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, vector_size, &B[0], NULL, &B_event);


		// Setup and execute the kernel
		cl::Kernel kernel_max = cl::Kernel(program, "min_kernel");
		kernel_max.setArg(0, buffer_A);
		//kernel_add.setArg(1, buffer_B);
		kernel_max.setArg(1, buffer_C);

		// ADDITION: Execute the multiplication kernel
		//cl::Kernel kernel_multadd = cl::Kernel(program, "multadd");
		//kernel_multadd.setArg(0, buffer_A);
		//kernel_multadd.setArg(1, buffer_B);
		//kernel_multadd.setArg(2, buffer_C);

		// COMMENT
		queue.enqueueNDRangeKernel(kernel_max, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &prof_event);


		// Copy the result from device to host
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0], NULL, &B_event);

		//std::cout << "A = " << A << std::endl;
		//std::cout << "B = " << B << std::endl;
		//std::cout << "C = " << C << std::endl;


		// Display the kernel + upload/download execution time
		cout << "\nKernel execution time for seperate kernels [ns]: " << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "\nUpload time for vector A [ns]: " << A_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - A_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;
		cout << "Upload time for vector B [ns]: " << B_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - B_event.getProfilingInfo<CL_PROFILING_COMMAND_START>() << endl;

		// Display profiling information
		cout << "\nProfiling information: " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << endl;

	}
	catch (cl::Error err) {
		cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
	}

	return 0;
};