cmake_minimum_required(VERSION 3.17)
project(libtorchtest)

set(CMAKE_CXX_STANDARD 14)

set(Torch_DIR /home/swing/libtorch/share/cmake/Torch/)  # 我的 libtorch 的路径

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)



add_executable(libtorchtest main.cpp)
target_link_libraries(libtorchtest
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        )