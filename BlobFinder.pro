QT -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += c++17

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    blobfinder.cpp

HEADERS += \
    blobfinder.h

# Default rules for deployment.
unix {
    target.path = $$[QT_INSTALL_PLUGINS]/generic
}
!isEmpty(target.path): INSTALLS += target

# Opencv
#opencv_version=420
#opencv_path=$$PWD/../../../opencv/opencv4.2
opencv_version=453
opencv_path=$$PWD/../../opencvlast/opencv-4.5.3/build
opencv_module_freetype=$$PWD/../../opencvlast/opencv_contrib-4.5.3/sources/modules/freetype
# freetype with harfbuzz 2.9.1 of 3rd party
opencv_contrib_path=$$PWD/../../opencvlast/opencv_contrib-4.5.3/build

INCLUDEPATH += $${opencv_path}/include
DEPENDPATH += $${opencv_path}/include
INCLUDEPATH += $${opencv_module_freetype}/include
DEPENDPATH += $${opencv_module_freetype}/include

CONFIG(debug, debug|release){
    win32: LIBS += -L$${opencv_path}/lib/ -lopencv_world$${opencv_version}d
    win32: LIBS += -L$${opencv_path}/lib/ -lopencv_videoio$${opencv_version}d
    win32: LIBS += -L$${opencv_contrib_path}/lib/Debug/ -lopencv_freetype$${opencv_version}d

    win32:!win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/opencv_world$${opencv_version}d.lib
    else:win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/libopencv_world$${opencv_version}d.a
    win32:!win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/opencv_videoio$${opencv_version}d.lib
    else:win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/libopencv_videoio$${opencv_version}d.a
    win32:!win32-g++: PRE_TARGETDEPS += $${opencv_contrib_path}/lib/Debug/opencv_freetype$${opencv_version}d.lib
    else:win32-g++: PRE_TARGETDEPS += $${opencv_contrib_path}/lib/Debug/libopencv_freetype$${opencv_version}d.a
} else {
    win32: LIBS += -L$${opencv_path}/lib/ -lopencv_world$${opencv_version}
    win32: LIBS += -L$${opencv_path}/lib/ -lopencv_videoio$${opencv_version}
    win32: LIBS += -L$${opencv_contrib_path}/lib/Release/ -lopencv_freetype$${opencv_version}

    win32:!win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/opencv_world$${opencv_version}.lib
    else:win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/libopencv_world$${opencv_version}.a
    win32:!win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/opencv_videoio$${opencv_version}.lib
    else:win32-g++: PRE_TARGETDEPS += $${opencv_path}/lib/libopencv_videoio$${opencv_version}.a
    win32:!win32-g++: PRE_TARGETDEPS += $${opencv_contrib_path}/lib/Release/opencv_freetype$${opencv_version}.lib
    else:win32-g++: PRE_TARGETDEPS += $${opencv_contrib_path}/lib/Release/libopencv_freetype$${opencv_version}.a
}
