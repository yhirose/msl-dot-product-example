#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define ANKERL_NANOBENCH_IMPLEMENT

#include <Metal/Metal.hpp>
#include <eigen3/Eigen/Core>
#include <vector>

#include "nanobench.h"

using namespace ankerl::nanobench;

//-----------------------------------------------------------------------------
// cpu_dot
//-----------------------------------------------------------------------------

void cpu_dot(const float* A, const float* B, float* OUT, size_t A_cols,
             size_t OUT_rows, size_t OUT_cols) {
  auto rows = OUT_rows;
  auto cols = OUT_cols;
  auto m = A_cols;

  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      float val = 0.0;
      for (size_t i = 0; i < m; i++) {
        val += A[A_cols * row + i] * B[OUT_cols * i + col];
      }
      OUT[OUT_cols * row + col] = val;
    }
  }
}

//-----------------------------------------------------------------------------
// class metal
//-----------------------------------------------------------------------------

class metal {
 public:
  template <typename U> struct releaser { void operator()(U* p) { p->release(); } };
  template <typename U> auto managed(U* p) { return std::shared_ptr<U>(p, releaser<U>()); }
  template <typename U> using managed_ptr = std::shared_ptr<U>;

  //---------------------------------------------------------------------------
  // constructor
  //---------------------------------------------------------------------------

  metal() : device_(managed(MTL::CreateSystemDefaultDevice())) {

    // MSL source code
    auto msl = R"(
      kernel void dot(
        device const void* A_bytes,
        device const void* B_bytes,
        device void*       OUT_bytes,
        constant uint32_t& A_cols,
        constant uint32_t& OUT_raws,
        constant uint32_t& OUT_cols,
        uint2              gid [[thread_position_in_grid]])
      {
        auto A = static_cast<device const float*>(A_bytes);
        auto B = static_cast<device const float*>(B_bytes);
        auto OUT = reinterpret_cast<device float*>(OUT_bytes);

        float val = 0.0;
        for (uint32_t i = 0; i < A_cols; i++) {
          val += A[A_cols * gid.y + i] * B[OUT_cols * i + gid.x];
        }

        OUT[OUT_cols * gid.y + gid.x] = val;
      }
    )";

    NS::Error* error = nullptr;

    // Compile MSL to make a Metal library
    auto src = NS::String::string(msl, NS::ASCIIStringEncoding);
    auto lib = managed(device_->newLibrary(src, nullptr, &error));

    // Create a `dot` pipeline state object
    auto str = NS::String::string("dot", NS::ASCIIStringEncoding);
    auto fn = managed(lib->newFunction(str));
    pso_dot_ = managed(device_->newComputePipelineState(fn.get(), &error));

    // Create a command queue
    queue_ = managed(device_->newCommandQueue());
  }

  //---------------------------------------------------------------------------
  // make_shared_buffer
  //---------------------------------------------------------------------------

  managed_ptr<MTL::Buffer> make_shared_buffer(NS::UInteger length) {
    return managed(device_->newBuffer(length, MTL::ResourceStorageModeShared));
  }

  //---------------------------------------------------------------------------
  // dot
  //---------------------------------------------------------------------------

  void dot(const managed_ptr<MTL::Buffer> A, const managed_ptr<MTL::Buffer> B,
           managed_ptr<MTL::Buffer> OUT, uint32_t A_cols, uint32_t OUT_rows,
           uint32_t OUT_cols) {
    auto commandBuffer = queue_->commandBuffer();

    auto computeEncoder = commandBuffer->computeCommandEncoder();

    computeEncoder->setComputePipelineState(pso_dot_.get());
    computeEncoder->setBuffer(A.get(), 0, 0);
    computeEncoder->setBuffer(B.get(), 0, 1);
    computeEncoder->setBuffer(OUT.get(), 0, 2);
    computeEncoder->setBytes(&A_cols, sizeof(uint32_t), 3);
    computeEncoder->setBytes(&OUT_rows, sizeof(uint32_t), 4);
    computeEncoder->setBytes(&OUT_cols, sizeof(uint32_t), 5);

    // Thread grid size
    auto gridSize = MTL::Size::Make(OUT_cols, OUT_rows, 1);

    // Efficient thread size
    auto w = pso_dot_->threadExecutionWidth();
    auto h = pso_dot_->maxTotalThreadsPerThreadgroup() / w;
    auto threadsSize = MTL::Size::Make(w, h, 1);

    computeEncoder->dispatchThreads(gridSize, threadsSize);
    computeEncoder->endEncoding();

    // Run
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
  }

 private:
  managed_ptr<MTL::Device> device_;
  managed_ptr<MTL::ComputePipelineState> pso_dot_;
  managed_ptr<MTL::CommandQueue> queue_;
};

//-----------------------------------------------------------------------------
// main
//-----------------------------------------------------------------------------

int main(void) {
  // Matrix size: A:{100, 1000} `dot` B:{1000, 100} = OUT:{1000, 100}
  size_t A_rows = 1000;
  size_t A_cols = 1000;
  size_t A_size = A_rows * A_cols;

  size_t B_rows = 1000;
  size_t B_cols = 100;
  size_t B_size = B_rows * B_cols;

  size_t OUT_rows = 1000;
  size_t OUT_cols = 100;
  size_t OUT_size = OUT_rows * OUT_cols;

  // CPU: Naive implementation
  {
    float A[A_size];
    float B[B_size];
    float OUT[OUT_size];

    Bench().minEpochIterations(10).run("CPU", [&] {

      cpu_dot(A, B, OUT, A_cols, OUT_rows, OUT_cols);

    });
  }

  // CPU: SIMD implementation
  {
    auto A = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(A_rows, A_cols);
    auto B = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>(B_rows, B_cols);
    auto O = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>();

    Bench().minEpochIterations(10).run("SIMD", [&] {

      O = A * B;

    });
  }

  // GPU
  {
    auto mtl = metal();

    // Byte buffers shareed on both CPU and GPU
    auto A = mtl.make_shared_buffer(A_size * sizeof(float));
    auto B = mtl.make_shared_buffer(B_size * sizeof(float));
    auto OUT = mtl.make_shared_buffer(OUT_size * sizeof(float));

    Bench().minEpochIterations(10).run( "GPU", [&] {

      mtl.dot(A, B, OUT, A_cols, OUT_rows, OUT_cols);

    });
  }
}
