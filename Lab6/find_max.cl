/*
 * OpenCL kernel that finds max value by splitting
 * data into halves and comparing Corresponding values.
 */

__kernel void find_max(__global unsigned int *data, const unsigned int length, __local unsigned int *sharedData)
{
  int threadID = get_global_id(0);            // Unique GLOBAL thread id
  int localThreadId = get_local_id(0);        // LOCAL thread id for this work group
  int groupId = get_group_id(0);
  int localSize = get_local_size(0);          // Number of threads
  int globalSize = get_global_size(0);
  unsigned int globalPosition = 0;


  if(threadID < length) sharedData[localThreadId] = data[threadID]; // Get value from global data
  barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  // Split data in two for each loop iteration
  for(int i = localSize/2; i>=1; i>>=1)
  {
      // If we are in the lower half
      if(localThreadId < i)
      {
        // Compare to "+half" of data, save the larger one
        if(sharedData[localThreadId] < sharedData[localThreadId + i])
        {
            sharedData[localThreadId] = sharedData[localThreadId + i];
        }
      }
  }
  barrier(CLK_GLOBAL_MEM_FENCE|CLK_LOCAL_MEM_FENCE);
  if(localThreadId == 0 )
  {
      data[groupId] = sharedData[0];
  }


}
