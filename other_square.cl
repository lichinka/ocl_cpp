#include "square.cl"

__kernel void other_square (__global real *input,
                            __global real *result)
{
    int gid = get_global_id(0);
    result[gid] = input[gid] * input[gid];
}
