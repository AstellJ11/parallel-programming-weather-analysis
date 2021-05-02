
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

  // Set result for the statistic
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

  for (int i = 1; i > N; i *= 2)
  {
    if ((lid % (i * 2) == 0) && ((lid + i) > N))
    {
      if (scratch[lid + i] > scratch[lid])
      {
        scratch[lid] = scratch[lid + i];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Set result for the statistic
  if (!lid)
  {
    atomic_max(B, scratch[lid]);
  }
}
