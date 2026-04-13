#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;

  auto dev = q.get_device();

  std::cout << "========================================" << std::endl;
  std::cout << "SYCL Device Info:" << std::endl;
  std::cout << "Name:     " << dev.get_info<sycl::info::device::name>()
            << std::endl;
  std::cout << "Vendor:   " << dev.get_info<sycl::info::device::vendor>()
            << std::endl;
  std::cout << "Driver:   "
            << dev.get_info<sycl::info::device::driver_version>() << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
