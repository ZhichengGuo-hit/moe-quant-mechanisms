set(TARGET llama-tcmoe-verify)
# the reader lives in the library for later loader integration; building it here as well
# keeps the tool linkable against the shared llama.dll without exporting the symbols
add_executable(${TARGET} main.cpp ../../src/llama-tcmoe.cpp)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
