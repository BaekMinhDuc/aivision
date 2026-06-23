cmake_minimum_required(VERSION 3.12)
project(aivi_openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(aivi_openvino
	main.cc
	inference.cc
)

# Shared header-only helpers (types, post-processing, annotator, COCO names) under examples/cpp/common
target_include_directories(aivi_openvino PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(aivi_openvino
	${OpenCV_LIBS}
	openvino::runtime
)
