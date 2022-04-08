FIND_PACKAGE(OpenCV REQUIRED)
SET(SRC_LIST main.cpp SemiGlobalMatching.cpp sgm_util.cpp stdafx.cpp)
ADD_EXECUTABLE(SemiGlobalMatching ${SRC_LIST})
#链接库要放在添加执行文件之后
TARGET_LINK_LIBRARIES(SemiGlobalMatching ${OpenCV_LIBS})
