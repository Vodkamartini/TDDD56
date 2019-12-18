/*
 * OpenCL kernel that performs the innermost
 * loop in the bitonic sorting algorithm.
 */

__kernel void bitonic(__global unsigned int *data, unsigned int j, unsigned int k)
{
  int gThreadID = get_global_id(0);   // Unique GLOBAL thread id --> Corresponds to i in CPU
  int groupID = get_group_id(0);      // Work group id

  int index = gThreadID^j;
  if(index > gThreadID)
  {
    if((gThreadID & k) == 0 && data[gThreadID]>data[index]) {
      unsigned int temp = data[gThreadID];
      data[gThreadID] = data[index];
      data[index] = temp;
    }
    if((gThreadID & k ) != 0 && data[gThreadID]<data[index]) {
      unsigned int temp = data[gThreadID];
      data[gThreadID] = data[index];
      data[index] = temp;
    }
  }
}
