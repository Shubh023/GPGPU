# Sequential lib
set(SRC 
    "lbp/lbp.cpp"
    "nearest-neighbor/nn_seq.cpp"
    "utils/io.cpp"
    "pred.cpp"
)

add_library(nn SHARED ${SRC})
target_include_directories(nn PUBLIC
    "."
    "lbp"
    "nearest-neighbor"
    "utils"    
)

# CUDA lib
add_library(cuda_nn SHARED "nearest-neighbor/nn_grid.cu"
                           "nearest-neighbor/nn_tiling.cu"
                           "lbp/lbp.cu")
                           
target_compile_features(cuda_nn PUBLIC cxx_std_14)
set_target_properties(cuda_nn PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_ARCHITECTURES 86)

# Create executable and link
add_executable(app main.cpp)
set_property(TARGET app PROPERTY CUDA_ARCHITECTURES 86)

target_link_libraries(app PUBLIC 
    "nn"
    "cuda_nn"
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
)

add_executable(bench bench.cpp)
set_property(TARGET bench PROPERTY CUDA_ARCHITECTURES 86)

target_link_libraries(bench PUBLIC 
    "nn"
    "cuda_nn"
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    benchmark
)