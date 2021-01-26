//#include <torch/torch.h>
#include <iostream>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/ATen.h>

void example1() {
  at::Tensor a = at::ones({10});
  at::Tensor b = at::ones({10});
  at::Tensor out = at::zeros({0});
  std::cout
    << "\n==========\n"
    << "example1()\n"
    << "=========="
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config.add_output(out)
    .add_input(a)
    .add_input(b);

  auto iter = iter_config.build();
  //======== End blog post code ===========
  std:: cout << "\nno output" << std::endl;
}

void example2() {
  at::Tensor a = at::randn({10});
  at::Tensor out = at::zeros({10});
  std::cout
    << "\n==========\n"
    << "example2()\n"
    << "=========="
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config.add_output(out)
    .add_input(a)

    // call if out was already allocated
    .resize_outputs(false)

    // call if inputs/outputs are of different types
    .check_all_same_dtype(false);

  auto iter = iter_config.build();

  // Copies data from input into output
  auto copy_loop = [&](char** data, const int64_t* strides, int64_t n) {
    // assume float data type for this example
    float* out_data_bytes = reinterpret_cast<float*>(data[0]);
    float* in_data_bytes = reinterpret_cast<float*>(data[1]);

    for (int i = 0; i < n; i++) {
      *out_data_bytes += *in_data_bytes;
      out_data_bytes++;
      in_data_bytes++;
    }
  };

  iter.for_each(copy_loop);
  //======== End blog post code ===========
  //
  std::cout << "\na:\n" << a << std::endl;
  std::cout << "\nout:\n" << out << std::endl;
}

void example3() {
  at::Tensor a = at::randn({10});
  at::Tensor b = at::randn({10});
  at::Tensor c = at::zeros({0});
  std::cout
    << "\n==========\n"
    << "example3()\n"
    << "=========="
    << std::endl;

  //======== Start blog post code =========
  at::TensorIteratorConfig iter_config;
  iter_config.add_output(c)
    .add_input(a)
    .add_input(b);

  auto iter = iter_config.build();
  at::native::cpu_kernel(iter, [] (float a, float b) -> float {
    return a + b;
  });
  //======== End blog post code ===========

  std::cout << "\na:\n" << a << std::endl;
  std::cout << "\nb:\n" << b << std::endl;
  std::cout << "\nc:\n" << c << std::endl;
}

int main() {
  at::manual_seed(0);
  example1();
  example2();
  example3();
}
