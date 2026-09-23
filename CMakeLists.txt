cmake_minimum_required(VERSION 3.20)

project(CosmicGradient VERSION 1.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Static CRT: an .aex should not depend on a redistributable being installed.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

if(MINGW)
    # Self-contained binaries: the .aex and the test tools must not depend on
    # the mingw runtime DLLs.
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

option(COSMIC_BUILD_PLUGIN "Build the After Effects .aex plug-in" ${WIN32})
option(COSMIC_BUILD_TESTS "Build the core tests and the preview tool" ON)

set(COSMIC_INSTALL_DIR "" CACHE PATH
    "Where 'cmake --install' copies CosmicGradient.aex (defaults to the shared Adobe plug-in folder on Windows)")

# ---------------------------------------------------------------------------
# Gradient core: no After Effects dependency, so it can be unit tested on any
# platform.
# ---------------------------------------------------------------------------
add_library(cosmic_core STATIC
    src/core/CosmicPipeline.cpp
    src/core/Palette.cpp
    src/core/Plan.cpp
    src/core/Pyramid.cpp
    src/core/Transfer.cpp
    src/core/UiModel.cpp
    src/gpu/GpuPipeline.cpp)

target_include_directories(cosmic_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")

if(MSVC)
    target_compile_options(cosmic_core PRIVATE /W4 /permissive- /Zc:__cplusplus)
else()
    target_compile_options(cosmic_core PRIVATE -Wall -Wextra)
endif()

set_target_properties(cosmic_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# After Effects plug-in
# ---------------------------------------------------------------------------
if(COSMIC_BUILD_PLUGIN)
    find_package(AfterEffectsSDK REQUIRED)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    # Stamped into the plug-in's UI so the loaded build can be identified.
    set(COSMIC_BUILD_ID "local")
    find_package(Git QUIET)
    if(GIT_EXECUTABLE)
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" rev-parse --short=7 HEAD
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            OUTPUT_VARIABLE _cosmic_git_sha
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_cosmic_git_sha)
            set(COSMIC_BUILD_ID "${_cosmic_git_sha}")
        endif()
    endif()
    message(STATUS "Cosmic Gradient build id: ${COSMIC_BUILD_ID}")

    set(COSMIC_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(COSMIC_PIPL_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_pipl.py")

    # The PiPL only changes when the plug-in's identity or flags change, so it is
    # generated at configure time and regenerated whenever the script changes.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${COSMIC_PIPL_SCRIPT}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${COSMIC_PIPL_SCRIPT}"
                --sdk-root "${AE_SDK_ROOT}"
                --output-dir "${COSMIC_GENERATED_DIR}"
                --version-major ${PROJECT_VERSION_MAJOR}
                --version-minor ${PROJECT_VERSION_MINOR}
                --version-bug ${PROJECT_VERSION_PATCH}
        RESULT_VARIABLE _pipl_result
        OUTPUT_VARIABLE _pipl_output
        ERROR_VARIABLE _pipl_error)
    if(NOT _pipl_result EQUAL 0)
        message(FATAL_ERROR "generate_pipl.py failed: ${_pipl_error}")
    endif()
    message(STATUS "Cosmic Gradient ${_pipl_output}")

    add_library(CosmicGradient MODULE
        src/plugin/CosmicEntry.cpp
        src/plugin/CosmicParams.cpp
        src/plugin/CosmicRender.cpp
        src/plugin/AeAdapters.cpp
        src/plugin/CudaDevice.cpp
        "${COSMIC_GENERATED_DIR}/CosmicGradient_PiPL.rc")

    target_link_libraries(CosmicGradient PRIVATE cosmic_core)
    target_include_directories(CosmicGradient PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/plugin"
        "${COSMIC_GENERATED_DIR}")
    target_include_directories(CosmicGradient SYSTEM PRIVATE ${AE_SDK_INCLUDE_DIRS})
    target_compile_definitions(CosmicGradient PRIVATE COSMIC_BUILD_ID="${COSMIC_BUILD_ID}")

    set_target_properties(CosmicGradient PROPERTIES
        PREFIX ""
        OUTPUT_NAME "CosmicGradient"
        SUFFIX ".aex")

    if(WIN32)
        # The SDK headers pick their platform from these.
        target_compile_definitions(CosmicGradient PRIVATE WIN_ENV _WINDOWS MSWindows NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()

    if(MSVC)
        target_compile_options(CosmicGradient PRIVATE /W4 /permissive- /Zc:__cplusplus)
        # Keep SDK header warnings out of our build log.
        target_compile_options(CosmicGradient PRIVATE /external:W0)
    else()
        target_compile_options(CosmicGradient PRIVATE -Wall -Wextra)
    endif()

    if(COSMIC_INSTALL_DIR)
        set(_cosmic_install_dir "${COSMIC_INSTALL_DIR}")
    elseif(WIN32)
        set(_cosmic_install_dir "C:/Program Files/Adobe/Common/Plug-ins/7.0/MediaCore/AB Tools")
    else()
        set(_cosmic_install_dir "${CMAKE_INSTALL_PREFIX}")
    endif()
    install(TARGETS CosmicGradient RUNTIME DESTINATION "${_cosmic_install_dir}"
                                   LIBRARY DESTINATION "${_cosmic_install_dir}")
endif()

# ---------------------------------------------------------------------------
# Tests and the offline preview tool
# ---------------------------------------------------------------------------
if(COSMIC_BUILD_TESTS)
    enable_testing()
    find_package(Threads REQUIRED)

    add_library(cosmic_test_support STATIC tests/TestSupport.cpp)
    target_include_directories(cosmic_test_support PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_link_libraries(cosmic_test_support PUBLIC cosmic_core Threads::Threads)

    add_executable(cosmic_tests tests/CoreTests.cpp)
    target_link_libraries(cosmic_tests PRIVATE cosmic_test_support)
    add_test(NAME core COMMAND cosmic_tests)

    add_executable(cosmic_preview tests/PreviewTool.cpp)
    target_link_libraries(cosmic_preview PRIVATE cosmic_test_support)

    # The CUDA kernels, run on the CPU against the CPU renderer.
    add_library(cosmic_gpu_emulator STATIC tests/GpuEmulator.cpp)
    target_link_libraries(cosmic_gpu_emulator PUBLIC cosmic_test_support)
    add_executable(cosmic_gpu_tests tests/GpuTests.cpp)
    target_link_libraries(cosmic_gpu_tests PRIVATE cosmic_gpu_emulator)
    add_test(NAME gpu COMMAND cosmic_gpu_tests)

    # The PTX the plug-in embeds must come from the kernels as they are now.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME ptx_up_to_date
                 COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/build_ptx.py" --check)
    endif()

    # Loads the built .aex and drives the After Effects command sequence.
    if(COSMIC_BUILD_PLUGIN AND WIN32)
        add_executable(cosmic_mock_host tests/MockHost.cpp)
        target_link_libraries(cosmic_mock_host PRIVATE cosmic_test_support)
        target_include_directories(cosmic_mock_host SYSTEM PRIVATE ${AE_SDK_INCLUDE_DIRS})
        target_compile_definitions(cosmic_mock_host PRIVATE WIN_ENV _WINDOWS MSWindows NOMINMAX WIN32_LEAN_AND_MEAN)
        add_dependencies(cosmic_mock_host CosmicGradient)

        # A stand-in nvcuda.dll: the plug-in's CUDA path, with kernels run by
        # the emulator, so the mock host can drive GPU renders without a GPU.
        add_library(fake_nvcuda SHARED tests/FakeCuda.cpp)
        target_link_libraries(fake_nvcuda PRIVATE cosmic_gpu_emulator)
        set_target_properties(fake_nvcuda PROPERTIES PREFIX "" OUTPUT_NAME "nvcuda")
        add_dependencies(cosmic_mock_host fake_nvcuda)
        add_test(NAME plugin COMMAND cosmic_mock_host $<TARGET_FILE:CosmicGradient> "" $<TARGET_FILE:fake_nvcuda>)
    endif()
endif()
