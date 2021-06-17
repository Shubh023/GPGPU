# Sequential lib
set(SRC "nearest-neighbor/nn_seq.cpp"
        "utils/io.cpp"
)

add_library(nn SHARED ${SRC})
target_include_directories(nn PUBLIC
    "lbp"
    "nearest-neighbor"
    "utils"    
)

# CUDA lib
add_library(cuda_nn SHARED "nearest-neighbor/nn.cu")
target_compile_features(cuda_nn PUBLIC cxx_std_14)
set_target_properties(cuda_nn PROPERTIES
                              CUDA_SEPARABLE_COMPILATION ON
                              CUDA_ARCHITECTURES 75)

# Create executable and link
add_executable(app main.cpp)
set_property(TARGET app PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(app PUBLIC 
    "nn"
    "cuda_nn"
    #${OpenCV_LIBS}
)