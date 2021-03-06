QT -= core gui

CONFIG += c++14

TARGET = PosterAugument
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

SOURCES += main.cpp \
    featurematcher.cpp \
    augumentor_cpu.cpp \
    augumentor.cpp

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


OPENCV_PATH = "C:/Code/opencv/opencv/build/install"
OPENCV_LIB_PATH = "$${OPENCV_PATH}/x64/vc14/lib/"
OPENCV_BIN_PATH = "$${OPENCV_PATH}/x64/vc14/bin/"
OPENCV_INCLUDE_PATH = "$${OPENCV_PATH}/include/"

BROFILER_PATH = "C:/Code/Brofiler-1.1.1/"

INCLUDEPATH += $${OPENCV_INCLUDE_PATH} \
               $${BROFILER_PATH}

LIBS += -L$${BROFILER_PATH} \
        -lProfilerCore64

debug {
    LIBS += -L$${OPENCV_LIB_PATH} \
    -lopencv_core320d \
    -lopencv_xfeatures2d320d \
    -lopencv_features2d320d \
    -lopencv_calib3d320d \
    -lopencv_video320d \
    -lopencv_highgui320d \
    -lopencv_videoio320d \
    -lopencv_imgcodecs320d \
    -lopencv_flann320d \
    -lopencv_imgproc320d

    OPENCV_BIN_PATH += "Debug/"

}

release {
LIBS += -L$${OPENCV_LIB_PATH} \
    -lopencv_core320 \
    -lopencv_xfeatures2d320 \
    -lopencv_features2d320 \
    -lopencv_calib3d320 \
    -lopencv_video320 \
    -lopencv_highgui320 \
    -lopencv_videoio320 \
    -lopencv_imgcodecs320 \
    -lopencv_flann320 \
    -lopencv_imgproc320

    OPENCV_BIN_PATH += "Release/"

}


HEADERS += \
    surf_image.h \
    featurematcher.h \
    augumentor_cpu.h \
    augumentor.h \
    profiling.h

