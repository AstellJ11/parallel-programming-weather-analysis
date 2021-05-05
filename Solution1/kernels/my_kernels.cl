
// Kernel to find the min value
kernel void min_reduce(global const int* A, global int* B, local int *scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    // Cache all values from global to local memory
    scratch[lid] = A[id];

    // Wait for all local threads to finish copying from global to local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2)
    {
        if ((lid % (i * 2) == 0) && ((lid + i) < N))
        {
            if (scratch[lid + i] < scratch[lid])
            {
                scratch[lid] = scratch[lid + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!lid)
    {
      atomic_min(B, scratch[lid]);
    }
}


// Kernel to find the max value
kernel void max_reduce(global const int* A, global int* B, local int *scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    // Cache all values from global to local memory
    scratch[lid] = A[id];

    // Wait for all local threads to finish copying from global to local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2)
    {
        if ((lid % (i * 2) == 0) && ((lid + i) < N))
        {
            if (scratch[lid + i] > scratch[lid])
            {
                scratch[lid] = scratch[lid + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!lid)
    {
        atomic_max(B, scratch[lid]);
    }
}


// Kernel to find the sum of the input vector
kernel void sum_reduce(global const int* A, global int* B, local int *scratch) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    // Cache all values from global to local memory
    scratch[lid] = A[id];

    // Wait for all local threads to finish copying from global to local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 1; i < N; i *= 2)
    {
        if ((lid % (i * 2) == 0) && ((lid + i) < N))
        {
            if (scratch[lid + i] += scratch[lid])
            {
                scratch[lid] = scratch[lid + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Less accurate if /100 here
    //scratch[lid] = scratch[lid] / 100;  

    if (!lid)
    {
        atomic_add(B, scratch[lid]);
    }
}


// Kernel to find the variance
// Variance = (sum((xi - mean)^2) / (n - 1))
kernel void variance_reduce(global const int* A, global int* B, local int *scratch, int mean, int input_size) {
    int id = get_global_id(0);
    int lid = get_local_id(0);
    int N = get_local_size(0);

    if (id < input_size) {  // Needed to ignore padding values
        // Cache all values from global to local memory while doing the first part of the variance calc.
 	    scratch[lid] = (A[id] - mean) * (A[id] - mean);

        // Wait for all local threads to finish copying from global to local memory
        barrier(CLK_LOCAL_MEM_FENCE);

        B[id] = scratch[lid];
    }
}


kernel void variance_sum_reduce(global const int* A, global int* B, local int *scratch) {
    int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

    // Cache all values from global to local memory
    scratch[lid] = A[id];

    // Wait for all local threads to finish copying from global to local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Same sum calculation as seen in sum_reduce
    for (int i = 1; i < N; i *= 2)
    {
        if ((lid % (i * 2) == 0) && ((lid + i) < N))
        {
            if (scratch[lid + i] += scratch[lid])
            {
                scratch[lid] = scratch[lid + i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Need to divide the result by 100 as x100 by 100 before parsing to kernel, but as formula is squared, 100^2 = 10000
	scratch[lid] = scratch[lid] / 10000;

	if (!lid) 
	{
		atomic_add(B, scratch[lid]);
	}
}
