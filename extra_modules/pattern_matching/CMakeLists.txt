set(the_description "Industrial pattern matching: Fast NCC, MatchTool NCC, Halcon-style shape matcher")
set(OPENCV_MODULE_IS_PART_OF_WORLD OFF)

ocv_define_module(pattern_matching opencv_core opencv_imgproc WRAP python)

ocv_module_include_directories("${CMAKE_CURRENT_LIST_DIR}/src")

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    ocv_target_link_libraries(${the_module} PUBLIC OpenMP::OpenMP_CXX)
    ocv_target_compile_definitions(${the_module} PRIVATE USE_OPENMP)
endif()

if(MSVC)
    ocv_target_compile_options(${the_module} PRIVATE /arch:AVX2 /fp:fast)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    ocv_target_compile_options(${the_module} PRIVATE -O3 -march=native -ffast-math)
endif()
