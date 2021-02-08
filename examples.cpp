#include <iostream>
#include <cassert>
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/AccumulateType.h>

void example1() {
  at::Tensor a = at::ones({10});
  at::Tensor b = at::ones({10});
  at::Tensor out = at::zeros({0});
  std::cout
    << "\n==========\n"
    << "example1:"
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config
    .add_output(out)
    .add_input(a)
    .add_input(b);

  auto iter = iter_config.build();
  //======== End blog post code ===========

  std::cout << "PASS" << std::endl;
}

void example2() {
  at::Tensor a = at::randn({10});
  at::Tensor out = at::randn({10});
  std::cout
    << "\n==========\n"
    << "example2:"
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config
    .add_output(out)
    .add_input(a)

    // call if output was already allocated
    .resize_outputs(false)

    // call if inputs/outputs have different types
    .check_all_same_dtype(false);

  auto iter = iter_config.build();

  // Copies data from input into output
  auto copy_loop = [](char** data, const int64_t* strides, int64_t n) {
    auto* out_data = data[0];
    auto* in_data = data[1];

    for (int64_t i = 0; i < n; i++) {
      // assume float data type for this example
      *reinterpret_cast<float*>(out_data) = *reinterpret_cast<float*>(in_data);
      out_data += strides[0];
      in_data += strides[1];
    }
  };

  iter.for_each(copy_loop);
  //======== End blog post code ===========

  assert((out == a).all().item<bool>());
  std::cout << "PASS" << std::endl;
}

void example3() {
  at::Tensor a = at::randn({10});
  at::Tensor b = at::randn({10});
  at::Tensor c = at::zeros({0});
  std::cout
    << "\n==========\n"
    << "example3:"
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config
    .add_output(c)
    .add_input(a)
    .add_input(b);

  auto iter = iter_config.build();

  // Element-wise add
  at::native::cpu_kernel(iter, [] (float a, float b) -> float {
    return a + b;
  });
  //======== End blog post code ===========

  assert((a + b).eq(c).all().item<bool>());
  std::cout << "PASS" << std::endl;
}

void example4() {
  std::cout
    << "\n==========\n"
    << "example4:"
    << std::endl;

  at::Tensor self = at::randn({7, 5});
  int64_t dim = 1;

  //======== Start blog post code =========
  // A cumulative sum's output is the same size as the input
  at::Tensor result = at::empty_like(self);

  at::TensorIteratorConfig iter_config;
  auto iter = iter_config
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dim=*/dim)
    .add_output(result)
    .add_input(self)
    .build();
  
  // Size of dimension to calculate the cumulative sum across
  int64_t self_dim_size = at::native::ensure_nonempty_size(self, dim);

  // These strides indicate number of memory-contiguous elements, not bytes,
  // between each successive element in dimension `dim`.
  auto result_dim_stride = at::native::ensure_nonempty_stride(result, dim);
  auto self_dim_stride = at::native::ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    // There are `n` individual vectors that span across dimension `dim`, so
    // `n` is equal to the number of elements in `self` divided by the size of
    // dimension `dim`.

    // These are the byte strides that separate each vector that spans across
    // dimension `dim`
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for (int64_t vector_idx = 0; vector_idx < n; ++vector_idx) {

      // Calculate cumulative sum for each element of the vector
      auto cumulative_sum = (at::acc_type<float, false>) 0;
      for (int64_t elem_idx = 0; elem_idx < self_dim_size; ++elem_idx) {
        const auto* self_data = reinterpret_cast<const float*>(self_data_bytes);
        auto* result_data = reinterpret_cast<float*>(result_data_bytes);
        cumulative_sum += self_data[elem_idx * self_dim_stride];
        result_data[elem_idx * result_dim_stride] = (float)cumulative_sum;
      }

      // Go to the next vector
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  iter.for_each(loop);
  //======== End blog post code ===========

  assert(result.eq(self.cumsum(dim)).all().item<bool>());
  std::cout << "PASS" << std::endl;
}

void example5() {
  std::cout
    << "\n==========\n"
    << "example5:"
    << std::endl;

  //======== Start blog post code =========
  at::Tensor self = at::randn({10, 10});
  int64_t dim = 1;
  bool keepdim = false;

  // `make_reduction` will allocate result Tensor for us, so we
  // can leave it undefined
  at::Tensor result;

  auto iter = at::native::make_reduction(
    "sum_reduce",
    result,
    self,
    dim,
    keepdim,
    self.scalar_type());

  // Sum reduce data from input into output
  auto sum_reduce_loop = [](char** data, const int64_t* strides, int64_t n) {
    auto* out_data = data[0];
    auto* in_data = data[1];

    *reinterpret_cast<float*>(out_data) = 0;

    for (int i = 0; i < n; i++) {
      // assume float data type for this example
      *reinterpret_cast<float*>(out_data) += *reinterpret_cast<float*>(in_data);
      in_data += strides[1];
    }
  };

  iter.for_each(sum_reduce_loop);
  //======== End blog post code ===========

  assert((self.sum(dim, keepdim) - result).abs().lt(0.00001).all().item<bool>());
  std::cout << "PASS" << std::endl;
}

int main() {
  at::manual_seed(0);
  example1();
  example2();
  example3();
  example4();
}
