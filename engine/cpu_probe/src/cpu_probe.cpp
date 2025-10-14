#include <cpuinfo.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (!cpuinfo_initialize()) return 1;

    std::string compiler = argc > 1 ? argv[1] : "gcc";
    bool        is_msvc  = (compiler == "msvc");

    std::map<std::string, std::vector<std::string>> gcc_flags = {
        {"SSE2", {"-msse2"}},
        {"SSE3", {"-msse3"}},
        {"AVX2", {"-mavx2"}},
        {"AVX512", {"-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq"}},
        {"POPCNT", {"-mpopcnt"}},
        {"PEXT", {"-mbmi2"}},
        {"NEON", {"-mfpu=neon"}},
        {"NEON_V8", {"-mfpu=neonv8"}},
        {"NEON_DOT", {"-mfpu=neon-dotprod"}},
        {"AVX512VNNI", {"-mavx512vnni"}},
        {"AVXVNNI", {"-mavxvnni"}}

    };
    std::map<std::string, std::vector<std::string>> msvc_flags = {
        {"SSE2", {"/arch:SSE2"}},     {"SSE3", {"/arch:SSE3"}},   {"AVX2", {"/arch:AVX2"}},
        {"AVX512", {"/arch:AVX512"}}, {"POPCNT", {"/arch:SSE2"}}, {"PEXT", {"/arch:AVX2"}}};
    auto& flags = is_msvc ? msvc_flags : gcc_flags;

    std::map<std::string, bool> cpu_features = {{"SSE2", cpuinfo_has_x86_sse2()},
                                                {"SSE3", cpuinfo_has_x86_sse3()},
                                                {"AVX2", cpuinfo_has_x86_avx2()},
                                                {"AVX512", cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512bw() &&
                                                               cpuinfo_has_x86_avx512vl() &&
                                                               cpuinfo_has_x86_avx512dq()},
                                                {"POPCNT", cpuinfo_has_x86_popcnt()},
                                                {"PEXT", cpuinfo_has_x86_bmi2()},
                                                {"NEON", cpuinfo_has_arm_neon()},
                                                {"NEON_V8", cpuinfo_has_arm_neon_v8()},
                                                {"NEON_DOT", cpuinfo_has_arm_neon_dot()},
                                                {"AVX512VNNI", cpuinfo_has_x86_avx512vnni()},
                                                {"AVXVNNI", cpuinfo_has_x86_avxvnni()},
                                                {"AMX", cpuinfo_has_x86_amx_tile()}};

    std::vector<std::string> enabled_flags;
    for (auto& f : cpu_features)
        if (f.second && flags.contains(f.first))
            for (const auto& flag : flags[f.first]) enabled_flags.push_back(flag);

    std::ofstream header("cpu_features.h");
    header << "#pragma once\n\n";
    for (auto& f : cpu_features) {

        if (f.second)
            std::cout << "Detected hardware feature : " << f.first << "\n";
        else
            std::cout << "Unavailablle hardware feature : " << f.first << "\n";
        header << "#define CHEPP_" << f.first << " " << (f.second ? 1 : 0) << "\n";
    }
    header.close();

    std::ofstream cmake_file("cpu_flags.cmake");
    cmake_file << "set(CPU_COMPILE_FLAGS";
    for (auto& f : enabled_flags) cmake_file << " " << f << "";
    cmake_file << ")\n";
    cmake_file.close();

    std::cout << "CPU detection done. Flags written to cpu_flags.cmake\n";
    return 0;
}
