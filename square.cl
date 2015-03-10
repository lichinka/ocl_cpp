#ifdef cl_khr_fp64
	#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#ifdef cl_amd_fp64
	#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif
typedef double	real;
typedef double2	real2;
typedef double4 real4;

#ifndef _MY_CONSTANT_
    #define _MY_CONSTANT_ 1
#endif

__kernel void square (__global real *input,
                      __global real *output)
{
    int gid_x = get_global_id(0);
    int gid_y = get_global_id(1);
    int size_x = get_global_size (0);
    int elem = gid_x + gid_y*size_x;
    output[elem] = input[elem] * input[elem];
}
