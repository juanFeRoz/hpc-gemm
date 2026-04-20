#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << "\n";
  return 0;
}
