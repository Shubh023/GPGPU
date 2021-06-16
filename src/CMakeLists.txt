set(SRC_LBP "nn.cpp")

add_library(lbp ${SRC_LBP})

target_include_directories(lbp PUBLIC .)

add_executable(app main.cpp)

target_link_libraries(app PUBLIC 
    lbp
    ${OpenCV_LIBS}
)