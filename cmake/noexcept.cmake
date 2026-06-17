add_library(noexcept INTERFACE)

target_compile_options(noexcept INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/GR->
        $<$<CXX_COMPILER_ID:MSVC>:/D_HAS_EXCEPTIONS=0>

        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-fno-rtti>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-fno-exceptions>
)