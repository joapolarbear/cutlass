/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/host_tensor_planar_complex.h"

#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/util/reference/device/gemm_planar_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include "cutlass/library/handle.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  std::string A;
  std::string B;
  std::string C;
  std::string accum;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({1024, 1024, 1024}),
    A("f32"),
    B("f32"),
    C("f32"),
    accum("f32"),
    batch_count(1),
    reference_check(true),
    iterations(20),
    alpha(1),
    beta() { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("batch", batch_count);

    cmd.get_cmd_line_argument("A", A);
    cmd.get_cmd_line_argument("B", B);
    cmd.get_cmd_line_argument("C", C);
    cmd.get_cmd_line_argument("accum", accum);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "18_cross_device_predict example\n\n"
        << "  This example uses the CUTLASS Library to execute GEMM computations.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement.\n\n"
        << "  --m <int>                   GEMM M dimension\n"
        << "  --n <int>                   GEMM N dimension\n"
        << "  --k <int>                   GEMM K dimension\n"
        << "  --A <string>                Matrix A data type\n"
        << "  --B <string>                Matrix B data type\n"
        << "  --C <string>                Matrix C data type\n"
        << "  --accum <string>            Accumulator data type\n"
        << "  --batch <int>               Number of GEMM operations executed in one batch\n"
        << "  --alpha <f32>               Epilogue scalar alpha\n"
        << "  --beta <f32>                Epilogue scalar beta\n\n"
        << "  --iterations <int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
        << "$ ./examples/18_cross_device_predict/18_cross_device_predict  --batch=7 --m=1024 --n=512 --k=1024 \\\n"
        << "     --alpha=2 --beta=0.707 --A=f16 --B=f16 --C=f16 --accum=f16 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count * 4;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Performance test environment for planar complex
class TestbedGEMM {
public:

  //
  // Data members
  //
  void const *ptr_A;
  void const *ptr_B;
  void const *ptr_C;
  void *ptr_D;

  cutlass::library::Handle handle;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;

  //
  // Methods
  //

  TestbedGEMM(
    Options const &options
  ): 
    problem_size(options.problem_size), batch_count(options.batch_count) {}

  template <typename T_A, typename T_B, typename T_C>
  void initialize(
      cutlass::DeviceAllocation<T_A> &tensor_A,
      cutlass::DeviceAllocation<T_B> &tensor_B,
      cutlass::DeviceAllocation<T_C> &tensor_C)
  {

    uint64_t seed = 1073;

    // Use small integers to simplify correctness checking
    int scope_max = 6;
    int scope_min = -6;

    cutlass::reference::device::BlockFillRandomUniform(
        tensor_A.get(), tensor_A.size(), seed, T_A(scope_max), T_A(scope_min), 0);

    cutlass::reference::device::BlockFillRandomUniform(
        tensor_B.get(), tensor_B.size(), seed * 2019, T_B(scope_max), T_B(scope_min), 0);

    cutlass::reference::device::BlockFillRandomUniform(
        tensor_C.get(), tensor_C.size(), seed * 2020, T_C(scope_max), T_C(scope_min), 0);
  }

  template <typename T>
  void reset(T tensor_ptr, std::string const &role) {
    if (role == "A") {
      tensor_ptr->reset(int64_t(problem_size.m()) * problem_size.k() * batch_count);
    } else if (role == "B") {
      tensor_ptr->reset(int64_t(problem_size.k()) * problem_size.n() * batch_count);
    } else if (role == "C") {
      tensor_ptr->reset(int64_t(problem_size.m()) * problem_size.n() * batch_count);
    } else if (role == "D") {
      tensor_ptr->reset(int64_t(problem_size.m()) * problem_size.n() * batch_count);
    }
  }

  void *allocate_matrix(
      std::string const &type_string,
      cutlass::library::NumericTypeID &type,
      std::string const &role)
  {
    void * ptr_;
    void * ret;
    uint64_t seed = 1073;

    // Use small integers to simplify correctness checking
    int scope_max = 6;
    int scope_min = -6;

    if (type_string == "f16") {
      ptr_ = new (cutlass::DeviceAllocation<cutlass::half_t>);
      reset((cutlass::DeviceAllocation<cutlass::half_t> *)(ptr_), role);
      cutlass::reference::device::BlockFillRandomUniform(
          ((cutlass::DeviceAllocation<cutlass::half_t>*)(ptr_))->get(), 
          ((cutlass::DeviceAllocation<cutlass::half_t>*)(ptr_))->size(), 
          seed, cutlass::half_t(scope_max), cutlass::half_t(scope_min), 0);
      ret = ((cutlass::DeviceAllocation<cutlass::half_t>*)(ptr_))->get();
      type = cutlass::library::NumericTypeID::kF16;
    }
    else if (type_string == "f32") {
      ptr_ = new (cutlass::DeviceAllocation<float>);
      reset((cutlass::DeviceAllocation<float> *)(ptr_), role);
      // std::cout << ((cutlass::DeviceAllocation<float> *)(ptr_))->size() << std::endl;
      cutlass::reference::device::BlockFillRandomUniform(
          ((cutlass::DeviceAllocation<float> *)(ptr_))->get(),
          ((cutlass::DeviceAllocation<float> *)(ptr_))->size(),
          seed, float(scope_max), float(scope_min), 0);
      ret = ((cutlass::DeviceAllocation<float> *)(ptr_))->get();
      // std::cout << ((cutlass::DeviceAllocation<float> *)(ptr_))->size() << std::endl;
      type = cutlass::library::NumericTypeID::kF32;
    }
    else {
      std::cout << "Invalid NumericTypeID: " << type_string 
            << ", should be one of [f16|f32]" << std::endl;
      exit(1);
    }
    return ret;
  }

  Result profile(Options const &options) {

    Result result;

    cutlass::library::NumericTypeID type_A;
    cutlass::library::NumericTypeID type_B;
    cutlass::library::NumericTypeID type_C;
    cutlass::library::NumericTypeID type_accum;

    // Allocate device memory for GEMM
    void *ptr_A = allocate_matrix(options.A, type_A, "A");
    void *ptr_B = allocate_matrix(options.B, type_B, "B");
    void *ptr_C = allocate_matrix(options.C, type_C, "C");
    void *ptr_D = allocate_matrix(options.C, type_C, "D");

    allocate_matrix(options.accum, type_accum, "null");
    // initialize(*ptr_A, *ptr_B, *ptr_C);

    int64_t batch_stride_A = int64_t(problem_size.m()) * problem_size.k();
    int64_t batch_stride_B = int64_t(problem_size.k()) * problem_size.n();
    int64_t batch_stride_C = int64_t(problem_size.m()) * problem_size.n();
    int64_t batch_stride_D = int64_t(problem_size.m()) * problem_size.n();

    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;

    int lda = LayoutA::packed({problem_size.m(), problem_size.k()}).stride(0);
    int ldb = LayoutB::packed({problem_size.k(), problem_size.n()}).stride(0);
    int ldc = LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);
    int ldd = LayoutC::packed({problem_size.m(), problem_size.n()}).stride(0);

    //
    // Construct events
    //

    cudaEvent_t events[2];

    for (auto & event : events) {
      result.error = cudaEventCreate(&event);
      if (result.error != cudaSuccess) {
        std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
        return -1;
      }
    }

    // Record an event at the start of a series of GEMMs
    result.error = cudaEventRecord(events[0]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    //
    // Run profiling loop
    //

    for (int iter = 0; iter < options.iterations; ++iter) {

      //
      // Execute the planar complex GEMM kernel via the CUTLASS Library's
      // dispatch routines.
      //
      // Note, for planar complex GEMM kernels, all numeric type arguments 
      // specify the data type of the base real types. These are understood to
      // apply to planar complex representations of matrices in memory and to complex<T>
      // structures for scalars.
      //
      // See tools/library/include/cutlass/library/handle.h for more details.
      //

      result.status = handle.gemm_universal(
          cutlass::library::GemmUniversalMode::kGemm,

          problem_size.m(), // GEMM M dimension
          problem_size.n(), // GEMM N dimension
          problem_size.k(), // GEMM K dimension

          type_accum, // Base data type of complex-valued accumulation
          type_accum, // Base data type of complex-valued alpha/beta scalars

          &options.alpha, // Pointer to alpha scalar, of type complex<T>

          type_A,                                       // Base data type of complex-valued A matrix
          cutlass::library::LayoutTypeID::kColumnMajor, // Layout of A matrix
          cutlass::library::ComplexTransform::kNone,    // Complex transformation on A matrix operand
          ptr_A,                                        // Pointer to A matrix in Global Memory
          lda,                                          // Leading dimension of A matrix

          type_B,                                       // Base data type of complex-valued B matrix
          cutlass::library::LayoutTypeID::kColumnMajor, // Layout of B matrix
          cutlass::library::ComplexTransform::kNone,    // Complex transformation on B matrix operand
          ptr_B,                                        // Pointer to B matrix in Global Memory
          ldb,                                          // Leading dimension of B matrix

          &options.beta, // Pointer to beta scalar, of type complex<T>

          type_C, // Base data type of complex valued C and D matrices

          ptr_C, // Pointer to C matrix
          ldc,   // Leading dimension of C matrix

          ptr_D, // Pointer to D matrix
          ldd,   // Leading dimension of D matrix

          batch_count, // Batch count or number of split-K slices

          batch_stride_A, // Batch stride of A operand
          batch_stride_B, // Batch stride of B operand
          batch_stride_C, // Batch stride of C operand
          batch_stride_D  // Batch stride of D operand

      );

      if (result.status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS internal error - configuration not supported" << std::endl;
        std::cerr << cutlass::cutlassGetStatusString(result.status) << std::endl;
        return result;
      }
    }
    
    //
    // Stop profiling loop
    //

    // Record an event when the GEMMs are complete
    result.error = cudaEventRecord(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = cudaEventSynchronize(events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
      return result;
    }

    // Compute average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)cudaEventDestroy(event);
    }

    if (handle.get_last_operation()) {
      std::cout << "Recently executed '" << handle.get_last_operation()->description().name << "'" << std::endl;
    }

    std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << " GFLOPs: " << result.gflops << std::endl;

    return result;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  //
  // This example uses mma.sync to directly access Tensor Cores to achieve peak performance.
  //
  // Volta Tensor Core operations are first available in CUDA 10.1 Toolkit.
  //
  // Turing Tensor Core operations are first available in CUDA 10.2 Toolkit.
  //

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major < 7) {
    std::cerr << "Volta Tensor Core operations must be run on a machine with compute capability at least 70."
              << std::endl;

    // Returning zero so this test passes on older architectures even though its actions are no-op.
    return 0;
  }
  else if (props.major == 7 && props.minor <= 2) {
    //
    // If running on the Volta architecture, at least CUDA 10.1 Toolkit is required to run this example.
    //
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
      std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

      // Returning zero so this test passes on older Toolkits even though its actions are no-op.
      return 0;
    }
  }
  else if (props.major == 7 && props.minor >= 5) {
    //
    // If running on the Turing architecture, at least CUDA 10.2 Toolkit is required to run this example.
    //
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
      std::cerr << "Turing Tensor Core operations must be compiled with CUDA 10.2 Toolkit or later." << std::endl;
    
      // Returning zero so this test passes on older Toolkits even though its actions are no-op.
      return 0;
    }
  }
  else {
    // NVIDIA Ampere Architecture GPUs (SM80 and later) are fully supported on CUDA 11 Toolkit and beyond.
    //
    // fall through
  }

  //
  // Parse options
  //

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Execute one problem size
  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  TestbedGEMM testbed(options);

  Result result = testbed.profile(options);

  return result.passed ? 0 : -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

