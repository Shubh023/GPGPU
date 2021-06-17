set(SRC "nearest-neighbor/nn_seq.cpp"
        "utils/io.cpp"
)

add_library(lbp ${SRC})

target_include_directories(lbp PUBLIC
    "lbp"
    "nearest-neighbor"
    "utils"    
)

add_executable(app main.cpp)

target_link_libraries(app PUBLIC 
    lbp
    ${OpenCV_LIBS}
)