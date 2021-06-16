set(SRC_LBP "lbp.cpp")
add_library(lbp ${SRC_LBP})

target_include_directories(lbp PUBLIC .)

find_package(OpenCV REQUIRED)

add_executable(app main.cpp)

target_link_libraries(app PRIVATE
    lbp
    #${OpenCV_LIBS}
    "opencv_core;opencv_imgcodecs;opencv_highgui"
)