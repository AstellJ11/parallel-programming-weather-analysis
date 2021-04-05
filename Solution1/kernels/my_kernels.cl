// COMMENT
kernel void max_temp_kernel(global const int* A, global const int* B, global int* C) {
	int id = get_global_id(0);
	C[id] = *max_element(temp.begin(), temp.end());
}
