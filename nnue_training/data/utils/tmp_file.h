//
// Created by paul on 9/17/25.
//

#ifndef CHEPP_TMP_FILE_H
#define CHEPP_TMP_FILE_H

#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

class TmpFile {
    std::filesystem::path path_;
    std::ofstream ofs_;
    std::ifstream ifs_;
    bool owns_file_ = true;

    static std::filesystem::path create_unique_path() {
#ifdef _WIN32
        char temp_path[MAX_PATH];
        if (!GetTempPathA(MAX_PATH, temp_path)) {
            throw std::runtime_error("GetTempPath failed");
        }

        char temp_file[MAX_PATH];
        if (!GetTempFileNameA(temp_path, "tmp", 0, temp_file)) {
            throw std::runtime_error("GetTempFileName failed");
        }

        return {std::string(temp_file)};
#else
        char tmpl[] = "/tmp/tmpfile_XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd < 0) throw std::runtime_error("mkstemp failed");
        close(fd);
        return {std::string(tmpl)};
#endif
    }

public:
    TmpFile() {
        path_ = create_unique_path();

        ofs_.open(path_, std::ios::binary | std::ios::trunc);
        if (!ofs_) throw std::runtime_error("Failed to open tmp file for writing");

        ifs_.open(path_, std::ios::binary);
        if (!ifs_) throw std::runtime_error("Failed to open tmp file for reading");

#ifndef _WIN32
        unlink(path_.c_str());
#endif
    }

    ~TmpFile() {
        ofs_.close();
        ifs_.close();

        if (owns_file_ && !path_.empty()) {
            std::remove(path_.c_str());
        }
    }

    TmpFile(const TmpFile&) = delete;
    TmpFile& operator=(const TmpFile&) = delete;

    TmpFile(TmpFile&& other) noexcept
        : path_(std::move(other.path_)),
          ofs_(std::move(other.ofs_)),
          ifs_(std::move(other.ifs_)),
          owns_file_(other.owns_file_) {
        other.owns_file_ = false;
    }

    TmpFile& operator=(TmpFile&& other) noexcept {
        if (this != &other) {
            ofs_.close();
            ifs_.close();

            if (owns_file_ && !path_.empty())
                std::remove(path_.c_str());

            path_ = std::move(other.path_);
            ofs_ = std::move(other.ofs_);
            ifs_ = std::move(other.ifs_);
            owns_file_ = other.owns_file_;
            other.owns_file_ = false;
        }
        return *this;
    }

    std::ofstream& ofstream() { return ofs_; }
    std::ifstream& ifstream() { return ifs_; }
    const std::filesystem::path& path() const { return path_; }
};

#endif // CHEPP_TMP_FILE_H
