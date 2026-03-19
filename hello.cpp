#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << "\n";
  q.submit([&](sycl::handler &cg) {
    auto os = sycl::stream{1024, 1024, cg};
    cg.parallel_for(10, [=](sycl::id<1> myid) {
      os << "Hello World! My ID is " << myid << "\n";
    });
  });
}
