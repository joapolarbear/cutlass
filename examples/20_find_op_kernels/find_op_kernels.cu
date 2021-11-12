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
#include "cutlass/library/util.h"

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

  int batch_count;
  float alpha;
  float beta;

  std::string A;
  std::string B;
  std::string C;
  std::string accum;

  int cc_major;
  int cc_minor;

  std::string operation;
  // GEMM options
  cutlass::gemm::GemmCoord problem_size;

  // Conv options
  std::string conv_kind;
  std::string iterator_algorithm;
  
  Options():
    help(false),
    operation("GEMM"),
    A("f32"),
    B("f32"),
    C("f32"),
    accum("f32"),
    batch_count(1),
    alpha(1),
    beta(),
    cc_major(-1),
    cc_minor(-1),
    problem_size({1024, 1024, 1024}),
    conv_kind("fprop"),
    iterator_algorithm("analytic")
    { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("batch", batch_count);

    cmd.get_cmd_line_argument("operation", operation);

    cmd.get_cmd_line_argument("A", A);
    cmd.get_cmd_line_argument("B", B);
    cmd.get_cmd_line_argument("C", C);
    cmd.get_cmd_line_argument("accum", accum);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("cc_major", cc_major);
    cmd.get_cmd_line_argument("cc_minor", cc_minor);

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("conv_kind", conv_kind);
    cmd.get_cmd_line_argument("iter_algo", iterator_algorithm);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "20_find_op_kernels example\n\n"
        << "  This example uses the CUTLASS Library to execute GEMM computations.\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage statement.\n\n"
        << "Common options:\n"
        << "  --operation <string>        GEMM or Conv\n"
        << "  --A <string>                Matrix A data type\n"
        << "  --B <string>                Matrix B data type\n"
        << "  --C <string>                Matrix C data type\n"
        << "  --accum <string>            Accumulator data type\n"
        << "  --batch <int>               Number of GEMM operations executed in one batch\n"
        << "  --cc_major <int>            Major part of target device compute capability. If not set, take the current device as the target device\n"
        << "  --cc_minor <int>            Minor part of target device compute capability. If not set, take the current device as the target device\n\n"
        << "GEMM options:\n"
        << "  --m <int>                   GEMM M dimension\n"
        << "  --n <int>                   GEMM N dimension\n"
        << "  --k <int>                   GEMM K dimension\n\n"
        << "  --alpha <f32>               Epilogue scalar alpha\n"
        << "  --beta <f32>                Epilogue scalar beta\n\n"
        << "Conv options:\n"
        << "  --conv_kind <string>        Convolutional kind, one"
        << " of fprop (forward propagation), dgrad (data gradient)"
        << " or wgrad (weight gradient)\n\n"
        << "  --iter_algo <string>        Convolutional iterator_algorithm,"
        << " one of analytic and optimized";

    out << "\n\nExamples:\n\n"
        << "$ ./examples/20_find_op_kernels/20_find_op_kernels  --batch=7 --m=1024 --n=512 --k=1024 \\\n"
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
      const cutlass::library::NumericTypeID &type,
      std::string const &role)
  {
    void * ptr_;
    void * ret;
    uint64_t seed = 1073;

    // Use small integers to simplify correctness checking
    int scope_max = 6;
    int scope_min = -6;

    if (type == cutlass::library::NumericTypeID::kF16) {
      ptr_ = new (cutlass::DeviceAllocation<cutlass::half_t>);
      reset((cutlass::DeviceAllocation<cutlass::half_t> *)(ptr_), role);
      cutlass::reference::device::BlockFillRandomUniform(
          ((cutlass::DeviceAllocation<cutlass::half_t>*)(ptr_))->get(), 
          ((cutlass::DeviceAllocation<cutlass::half_t>*)(ptr_))->size(), 
          seed, cutlass::half_t(scope_max), cutlass::half_t(scope_min), 0);
      ret = ((cutlass::DeviceAllocation<cutlass::half_t>*)(ptr_))->get();
    } else if (type == cutlass::library::NumericTypeID::kF32) {
      ptr_ = new (cutlass::DeviceAllocation<float>);
      reset((cutlass::DeviceAllocation<float> *)(ptr_), role);
      // std::cout << ((cutlass::DeviceAllocation<float> *)(ptr_))->size() << std::endl;
      cutlass::reference::device::BlockFillRandomUniform(
          ((cutlass::DeviceAllocation<float> *)(ptr_))->get(),
          ((cutlass::DeviceAllocation<float> *)(ptr_))->size(),
          seed, float(scope_max), float(scope_min), 0);
      ret = ((cutlass::DeviceAllocation<float> *)(ptr_))->get();
      // std::cout << ((cutlass::DeviceAllocation<float> *)(ptr_))->size() << std::endl;
    } else {
      std::cout << "Invalid NumericTypeID: " 
          << cutlass::library::to_string(type, true)
          << ", should be one of [f16|f32]" << std::endl;
      exit(1);
    }
    return ret;
  }

  void parse_type_layout(
    const std::string &istring,
    cutlass::library::NumericTypeID &type,
    cutlass::library::LayoutTypeID &layout)
  {
    size_t s_idx = 0;
    size_t d_idx = std::string::npos;
    d_idx = istring.find_first_of(",", s_idx);
    if (d_idx != std::string::npos) {
      size_t end_idx = d_idx;
      type = cutlass::library::from_string<cutlass::library::NumericTypeID>(istring.substr(s_idx, end_idx));
      s_idx = end_idx + 1;
      end_idx = istring.size();
      layout = cutlass::library::from_string<cutlass::library::LayoutTypeID>(istring.substr(s_idx, end_idx));
    } else {
      size_t end_idx = istring.size();
      type = cutlass::library::from_string<cutlass::library::NumericTypeID>(istring.substr(s_idx, end_idx));
    }    
  }

  Result query_gemm(Options const &options, const int cc_major, const int cc_minor) {

    Result result;

    cutlass::library::NumericTypeID type_A = cutlass::library::NumericTypeID::kF32;
    cutlass::library::NumericTypeID type_B = cutlass::library::NumericTypeID::kF32;
    cutlass::library::NumericTypeID type_C = cutlass::library::NumericTypeID::kF32;
    cutlass::library::NumericTypeID type_accum = cutlass::library::NumericTypeID::kF32;

    cutlass::library::LayoutTypeID layout_A = cutlass::library::LayoutTypeID::kColumnMajor;
    cutlass::library::LayoutTypeID layout_B = cutlass::library::LayoutTypeID::kColumnMajor;
    cutlass::library::LayoutTypeID layout_C = cutlass::library::LayoutTypeID::kColumnMajor;
    cutlass::library::LayoutTypeID layout_accum = cutlass::library::LayoutTypeID::kColumnMajor;

    parse_type_layout(options.A, type_A, layout_A);
    parse_type_layout(options.B, type_B, layout_B);
    parse_type_layout(options.C, type_C, layout_C);
    parse_type_layout(options.accum, type_accum, layout_accum);

    // Allocate device memory for GEMM
    void *ptr_A = allocate_matrix(type_A, "A");
    void *ptr_B = allocate_matrix(type_B, "B");
    void *ptr_C = allocate_matrix(type_C, "C");
    void *ptr_D = allocate_matrix(type_C, "D");
    allocate_matrix(type_accum, "null");


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

    auto operation = handle.find_gemm_kernel(
        cutlass::library::GemmUniversalMode::kGemm,

        problem_size.m(), // GEMM M dimension
        problem_size.n(), // GEMM N dimension
        problem_size.k(), // GEMM K dimension

        type_accum, // Base data type of complex-valued accumulation
        type_accum, // Base data type of complex-valued alpha/beta scalars

        type_A,                                    // Base data type of complex-valued A matrix
        layout_A,                                  // Layout of A matrix
        cutlass::library::ComplexTransform::kNone, // Complex transformation on A matrix operand
        ptr_A,                                     // Pointer to A matrix in Global Memory
        lda,                                       // Leading dimension of A matrix

        type_B,                                    // Base data type of complex-valued B matrix
        layout_B,                                  // Layout of B matrix
        cutlass::library::ComplexTransform::kNone, // Complex transformation on B matrix operand
        ptr_B,                                     // Pointer to B matrix in Global Memory
        ldb,                                       // Leading dimension of B matrix

        type_C, // Base data type of complex valued C and D matrices

        ptr_C, // Pointer to C matrix
        ldc,   // Leading dimension of C matrix

        ptr_D,    // Pointer to D matrix
        ldd,      // Leading dimension of D matrix
        cc_major, /// Compute capability major
        cc_minor  /// Compute capability minor
    );

    if (operation) {
      std::cout << "Recently executed '" << operation->description().name << "'" << std::endl;
    }
    return result;
  }

  Result query_conv(Options const &options, const int cc_major, const int cc_minor) {

    Result result;

    cutlass::library::NumericTypeID type_A = cutlass::library::NumericTypeID::kF32;
    cutlass::library::NumericTypeID type_B = cutlass::library::NumericTypeID::kF32;
    cutlass::library::NumericTypeID type_C = cutlass::library::NumericTypeID::kF32;
    cutlass::library::NumericTypeID type_accum = cutlass::library::NumericTypeID::kF32;

    cutlass::library::LayoutTypeID layout_A = cutlass::library::LayoutTypeID::kTensorNHWC;
    cutlass::library::LayoutTypeID layout_B = cutlass::library::LayoutTypeID::kTensorNHWC;
    cutlass::library::LayoutTypeID layout_C = cutlass::library::LayoutTypeID::kTensorNHWC;
    cutlass::library::LayoutTypeID layout_accum = cutlass::library::LayoutTypeID::kTensorNHWC;

    parse_type_layout(options.A, type_A, layout_A);
    parse_type_layout(options.B, type_B, layout_B);
    parse_type_layout(options.C, type_C, layout_C);
    parse_type_layout(options.accum, type_accum, layout_accum);

    cutlass::library::ConvKind conv_kind;
    conv_kind = cutlass::library::from_string<cutlass::library::ConvKind>(options.conv_kind);

    cutlass::library::IteratorAlgorithmID iterator_algorithm;
    iterator_algorithm = cutlass::library::from_string<cutlass::library::IteratorAlgorithmID>(options.iterator_algorithm);

    auto operation = handle.find_conv_kernel(
        cutlass::library::OperationKind::kConv2d,
        conv_kind,

        type_A,                               // Base data type of complex-valued A matrix
        layout_A,                             // Layout of A matrix

        type_B,                               // Base data type of complex-valued B matrix
        layout_B,                             // Layout of B matrix

        type_C,                               // Base data type of complex valued C and D matrices
        layout_C,                             // Layout of C matrix

        type_accum,                           // Base data type of complex-valued accumulation
        type_accum,                           // Base data type of complex-valued alpha/beta scalars

        iterator_algorithm,

        cc_major,                             /// Compute capability major
        cc_minor                              /// Compute capability minor
    );

    if (operation) {
      std::cout << "Recently executed '" << operation->description().name << "'" << std::endl;
    }
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

  int cc_major, cc_minor;

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help)
  {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Execute one problem size
  if (!options.valid())
  {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }
  
  if (options.cc_major >= 0 && options.cc_minor >= 0) {
    cc_major = options.cc_major;
    cc_minor = options.cc_minor;
  } else {
    cudaDeviceProp props;
    cudaError_t error = cudaGetDeviceProperties(&props, 0);
    if (error != cudaSuccess) {
      std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
      return -1;
    }

    cc_major = props.major;
    cc_minor = props.minor;
  }
    
  if (cc_major < 7) {
    std::cerr << "Volta Tensor Core operations must be run on a machine with compute capability at least 70."
              << std::endl;

    // Returning zero so this test passes on older architectures even though its actions are no-op.
    return 0;
  }
  else if (cc_major == 7 && cc_minor <= 2) {
    //
    // If running on the Volta architecture, at least CUDA 10.1 Toolkit is required to run this example.
    //
    if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 1))) {
      std::cerr << "Volta Tensor Core operations must be compiled with CUDA 10.1 Toolkit or later." << std::endl;

      // Returning zero so this test passes on older Toolkits even though its actions are no-op.
      return 0;
    }
  }
  else if (cc_major == 7 && cc_minor >= 5) {
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

  TestbedGEMM testbed(options);
  Result result;
  if (options.operation == "gemm") {
    result = testbed.query_gemm(options, cc_major, cc_minor);
  } else if (options.operation == "conv2d_fprop") {
    result = testbed.query_conv(options, cc_major, cc_minor);
  } else {
    std::cout << "Invalid operation is given: " << options.operation << std::endl;
  }

  return result.passed ? 0 : -1;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

