#include <iostream>
#include <cassert>
#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/ReduceOpsUtils.h>

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
  at::Tensor out = at::zeros({10});
  std::cout
    << "\n==========\n"
    << "example2:"
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config
    .add_output(out)
    .add_input(a)

    // call if out was already allocated
    .resize_outputs(false)

    // call if inputs/outputs are of different types
    .check_all_same_dtype(false);

  auto iter = iter_config.build();

  // Copies data from input into output
  auto copy_loop = [&](char** data, const int64_t* strides, int64_t n) {

    // assume float data type for this example
    float* out_data = reinterpret_cast<float*>(data[0]);
    float* in_data = reinterpret_cast<float*>(data[1]);

    for (int i = 0; i < n; i++) {
      *out_data += *in_data;
      out_data++;
      in_data++;
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
  at::Tensor self = at::randn({10});
  std::cout
    << "\n==========\n"
    << "example4:"
    << std::endl;

  //======== Start blog post code =========
  // `make_reduction` will allocate result Tensor for us, so we
  // can leave it undefined
  at::Tensor result;

  int64_t dim = 0;
  bool keepdim = false;
  auto iter = at::native::make_reduction(
    "sum_reduce",
    result,
    self,
    dim,
    keepdim,
    self.scalar_type());

  // Sum reduce data from input into output
  auto sum_reduce_loop = [&](char** data, const int64_t* strides, int64_t n) {
    // assume float data type for this example
    float* out_data = reinterpret_cast<float*>(data[0]);
    float* in_data = reinterpret_cast<float*>(data[1]);

    *out_data = 0;
    for (int i = 0; i < n; i++) {
      *out_data += *in_data;
      in_data++;
    }
  };

  iter.for_each(sum_reduce_loop);
  //======== End blog post code ===========

  assert((self.sum(dim, keepdim) - result).abs().lt(0.00001).all().item<bool>());
  std::cout << "PASS" << std::endl;
}

int main() {
  example1();
  example2();
  example3();
  example4();
}
