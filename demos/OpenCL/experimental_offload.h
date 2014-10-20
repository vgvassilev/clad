// Include OpenCL headers
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <strings.h>
#else
#include <CL/cl.h>
#include <stdio.h>
#include <memory.h>
#endif

// Offload template
template<typename Func>
float experimental_offload(Func func) {
  return func();
}


// Offload conceptual implementation
const char* rosenbrock_func_cl[] = {"\
static float rosenbrock_func_darg0(float x, float y) {\n\
  return 2.F * (x - 1.F) - 400.F * x * (y - x * x);\n\
}\n\
static float rosenbrock_func_darg1(float x, float y) {\n\
  return 200.F * (y - x * x);\n\
}\n\
\n\
kernel void rosenbrock_pfor_body(global float4* x, global float4* temp) {\n\
//printf(\"gsize=%d\\n\", 1 /*get_global_size(0)/4096*/);\n\
//printf(\"lsize=%d\\n\", 2 /*get_local_size(0)/4096*/);\n\
  size_t i = get_global_id(0);\n\
  float4 one = (float4)(rosenbrock_func_darg0(x[i].s0, x[i].s1),rosenbrock_func_darg0(x[i].s1, x[i].s2),rosenbrock_func_darg0(x[i].s2, x[i].s3),rosenbrock_func_darg0(x[i].s3, x[i + 1].s0));\n\
  float4 two = (float4)(rosenbrock_func_darg1(x[i].s0, x[i].s1),rosenbrock_func_darg1(x[i].s1, x[i].s2),rosenbrock_func_darg1(x[i].s2, x[i].s3),rosenbrock_func_darg1(x[i].s3, x[i + 1].s0));\n\
  temp[i] = one + two;\n\
}\n\
\n\
kernel void rosenbrock_pfor_reduction(global float4* data, local float4* partial_sums) {\n\
   size_t lid = get_local_id(0);\n\
   size_t lsize = get_local_size(0);\n\
//printf(\"gsize=%d\\n\", get_global_size(0));\n\
//printf(\"lsize=%d\\n\", lsize);\n\
//for (int j=0; j<lsize; j++) printf(\"%f,%f,%f,%f,\",data[j].s0,data[j].s1,data[j].s2,data[j].s3);\n\
//printf(\"\\n\");\n\
\n\
   partial_sums[lid] = data[get_global_id(0)];\n\
   barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
   for (int i = lsize/2; i>0; i >>= 1) {\n\
      if (lid < i) {\n\
         partial_sums[lid] += partial_sums[lid + i];\n\
      }\n\
      barrier(CLK_LOCAL_MEM_FENCE);\n\
   }\n\
\n\
   if (lid == 0) {\n\
      data[get_group_id(0)] = partial_sums[0];\n\
   }\n\
}\n\
\n\
kernel void rosenbrock_pfor_reduction_end(global float4* data, local float4* partial_sums, global float* sum) {\n\
   size_t lid = get_local_id(0);\n\
   size_t lsize = get_local_size(0);\n\
//printf(\"lsize=%d\\n\", lsize);\n\
//for (int j=0; j<lsize; j++) printf(\"%f,%f,%f,%f,\",data[j].s0,data[j].s1,data[j].s2,data[j].s3);\n\
//printf(\"\\n\");\n\
\n\
   partial_sums[lid] = data[get_local_id(0)];\n\
   barrier(CLK_LOCAL_MEM_FENCE);\n\
\n\
   for (int i = lsize/2; i>0; i >>= 1) {\n\
      if (lid < i) {\n\
         partial_sums[lid] += partial_sums[lid + i];\n\
      }\n\
      barrier(CLK_LOCAL_MEM_FENCE);\n\
   }\n\
\n\
   if (lid == 0) {\n\
      *sum = dot(partial_sums[0], (float4)(1.F));\n\
      //printf(\"(%f)\",*sum);\n\
   }\n\
}\n\0",
"\0"};


// Helper functions

inline cl_int check(cl_int err) {
  if (err!=CL_SUCCESS) printf("Error: %d\n", err);
  return err;
}

/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
  cl_platform_id platforms[10];
  cl_platform_id platform;
  cl_device_id dev;

  /* Identify a platform */
  check(clGetPlatformIDs(1, platforms, NULL));
  platform=platforms[0];

  /* Access a device */
  if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL) == CL_DEVICE_NOT_FOUND) {
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
  }
  //printf("result_dev=%p\n", dev);

  char* value;
  size_t valueSize;
  clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize+1);
  check(clGetDeviceInfo(dev, CL_DEVICE_NAME, valueSize, value, NULL));
  printf("Device=%s\n", value);
  clGetDeviceInfo(dev, CL_DEVICE_VENDOR, 0, NULL, &valueSize);
  value = (char*) malloc(valueSize+1);
  check(clGetDeviceInfo(dev, CL_DEVICE_VENDOR, valueSize, value, NULL));
  printf("Vendor=%s\n", value);

  return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char** source) {
  cl_program program;

  /* Create program from file */
  program = clCreateProgramWithSource(ctx, 1, source, NULL, NULL);

  /* Build program */
  //if (check(clBuildProgram(program, 0, NULL, "", NULL, NULL))) {
  if (check(clBuildProgram(program, 0, NULL, "-cl-denorms-are-zero -cl-finite-math-only -cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-fast-relaxed-math", NULL, NULL))) {
      /* Find size of log and print to std output */
      char *program_log;
      size_t log_size;
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   };

  return program;
}

/* OpenCL global structures */
cl_device_id device;
cl_context context;
cl_program program;
cl_kernel kernel_body;
cl_kernel kernel_reduction;
cl_kernel kernel_reduction_end;
cl_command_queue queue;
size_t local_size;
cl_event start_event, end_event;
cl_ulong time_start, time_end, total_time;

float *temp = NULL;
cl_mem data_buffer = 0, temp_buffer = 0, sum_buffer = 0;

void init_experimental_offload() {
  /* Create device and determine local size */
  device = create_device();
  check(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(local_size), &local_size, NULL));
#ifdef __APPLE__
  printf("result_max-work-group-size=%zu\n", local_size);
#else
  printf("result_max-work-group-size=%d\n", local_size);
#endif

  /* Create a context */
  context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  //printf("result_context=%p\n", context);

  /* Build program */
  program = build_program(context, device, rosenbrock_func_cl);
  //printf("result_program=%p\n", program);

  /* Create a command queue */
  queue = clCreateCommandQueue(context, device, 0 /* | CL_QUEUE_PROFILING_ENABLE*/, NULL);
  //printf("result_queue=%p\n", queue);

  /* Create a kernels */
  kernel_body = clCreateKernel(program, "rosenbrock_pfor_body", NULL);
  kernel_reduction = clCreateKernel(program, "rosenbrock_pfor_reduction", NULL);
  kernel_reduction_end = clCreateKernel(program, "rosenbrock_pfor_reduction_end", NULL);
  //printf("result_kernel_body=%p\n", kernel_body);
  //printf("result_kernel_reduction=%p\n", kernel_reduction);
  //printf("result_kernel_reduction_end=%p\n", kernel_reduction_end);
}

float experimental_offloaded(const float data[], const int data_size) {
  /* Data and buffers */
  size_t global_size_body, global_size_reduction;
  cl_event start_event, end_event;

  float sum = 0.0f;

  //printf("param_data_size=%d\n", data_size);

  /* Create data buffer */
  if (temp == NULL) {
    /* Allocate and initialize output arrays */
    temp = (float*)malloc(data_size * sizeof(float));
    //printf("result_num_groups=%d\n", num_groups);
    //scalar_sum = (float*)malloc(num_groups * sizeof(float));
    //for(int i=0; i<num_groups; i++) {
    //  scalar_sum[i] = 0.0f;
    //}

    //data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, data_size * sizeof(float), (void *)data, NULL);
    data_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, data_size * sizeof(float), NULL, NULL);
    temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE /*| CL_MEM_HOST_NO_ACCESS */ | CL_MEM_USE_HOST_PTR, data_size * sizeof(float), temp, NULL);
    sum_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, NULL);
    //printf("result_data_buffer=%p\n", data_buffer);
    //printf("result_temp_buffer=%p\n", temp_buffer);
    //printf("result_sum_buffer=%p\n", sum_buffer);
  }
  memset(temp, 0, data_size * sizeof(float));
  //for (int i = 0; i<data_size; i++) {
  //  temp[i] = 0.0f;
  //}

  /* Write input data to buffer */
  //check(clEnqueueWriteBuffer(queue, data_buffer, CL_TRUE, 0, data_size * sizeof(float), data, 0, NULL, NULL));
  check(clEnqueueWriteBuffer(queue, data_buffer, CL_FALSE, 0, data_size * sizeof(float), data, 0, NULL, NULL));
  //printf("%f\n", sum);


  /* Create kernel arguments */
  global_size_body = (data_size/4)-1;
  check(clSetKernelArg(kernel_body, 0, sizeof(cl_mem), &data_buffer));
  check(clSetKernelArg(kernel_body, 1, sizeof(cl_mem), &temp_buffer));
  //printf("result_global_size_body=%zu\n", global_size_body);
  /* Enqueue parallel body kernel */
  check(clEnqueueNDRangeKernel(queue, kernel_body, 1, NULL, &global_size_body, NULL, 0, NULL, NULL));
/*
  check(clEnqueueReadBuffer(queue, temp_buffer, CL_TRUE, 0, data_size * sizeof(float), temp, 0, NULL, NULL));
  for (int i=0; i<16; i++) {
    printf("%f,", temp[i]);
  }
  printf("\n");
*/

  global_size_reduction = data_size/4;
  //printf("result_global_size_reduction=%zu\n", global_size_reduction);
  /* Perform successive stages of the reduction */
  while (global_size_reduction/local_size > local_size) {
    global_size_reduction = global_size_reduction/local_size;
    check(clSetKernelArg(kernel_reduction, 0, sizeof(cl_mem), &temp_buffer));
    check(clSetKernelArg(kernel_reduction, 1, local_size * 4 * sizeof(float), NULL));
//printf("9===%zu/%zu\n", global_size_reduction, local_size);
    check(clEnqueueNDRangeKernel(queue, kernel_reduction, 1, NULL, &global_size_reduction, NULL /*&local_size*/, 0, NULL, NULL));
  }
/*
  check(clEnqueueReadBuffer(queue, temp_buffer, CL_TRUE, 0, data_size * sizeof(float), temp, 0, NULL, NULL));
  for (int i=0; i<16; i++) {
    printf("%f,", temp[i]);
  }
  printf("\n");
*/
  /* Perform completion stages of the reduction */
  global_size_reduction = global_size_reduction/local_size;
  check(clSetKernelArg(kernel_reduction_end, 0, sizeof(cl_mem), &temp_buffer));
  check(clSetKernelArg(kernel_reduction_end, 1, local_size * 4 * sizeof(float), NULL));
  check(clSetKernelArg(kernel_reduction_end, 2, sizeof(cl_mem), &sum_buffer));
  check(clEnqueueNDRangeKernel(queue, kernel_reduction_end, 1, NULL, &global_size_reduction, NULL, 0, NULL, NULL));

  /* Finish processing the queue and get profiling information */
  check(clFinish(queue));

/*
  check(clEnqueueReadBuffer(queue, temp_buffer, CL_TRUE, 0, data_size * sizeof(float), temp, 0, NULL, NULL));
  for (int i=0; i<data_size; i++) {
    printf("%f,", temp[i]);
  }
  printf("\n");
*/

  /* Read the result */
  check(clEnqueueReadBuffer(queue, sum_buffer, CL_TRUE, 0, sizeof(float), &sum, 0, NULL, NULL));
  //printf("%f\n", sum);

  //clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  //clGetEventProfilingInfo(start_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  ////clGetEventProfilingInfo(end_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  //total_time = time_end - time_start;
  //printf("Total time = %llu\n", total_time);

//  check(clReleaseMemObject(sum_buffer));
//  check(clReleaseMemObject(temp_buffer));
//  check(clReleaseMemObject(data_buffer));
//  free(temp);

  return sum;
}

void done_experimental_offload() {
  /* Deallocate resources */
  if (temp != NULL) {
    check(clReleaseMemObject(sum_buffer));
    check(clReleaseMemObject(temp_buffer));
    check(clReleaseMemObject(data_buffer));
    free(temp);
  }
  clReleaseKernel(kernel_body);
  clReleaseKernel(kernel_reduction);
  clReleaseKernel(kernel_reduction_end);
  clReleaseCommandQueue(queue);
  clReleaseProgram(program);
  clReleaseContext(context);
}
