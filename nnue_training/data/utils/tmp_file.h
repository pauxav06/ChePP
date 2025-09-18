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
#include <io.h>
#include <fcntl.h>
#else
#include <unistd.h>
#endif

class TmpFile {
    std::filesystem::path path_;
    std::fstream fs_;
    bool owns_file_ = true;

    static std::filesystem::path create_unique_path() {
#ifdef _WIN32
        wchar_t temp_path[MAX_PATH];
        if (!GetTempPathW(MAX_PATH, temp_path)) {
            throw std::runtime_error("GetTempPath failed");
        }

        wchar_t temp_file[MAX_PATH];
        if (!GetTempFileNameW(temp_path, L"tmp", 0, temp_file)) {
            throw std::runtime_error("GetTempFileName failed");
        }

        return {temp_file};
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
#ifdef _WIN32
        path_ = create_unique_path();

        HANDLE hFile = CreateFileW(
            path_.c_str(),
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
            nullptr
        );
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("CreateFile failed");
        }

        int fd = _open_osfhandle((intptr_t)hFile, _O_RDWR | _O_BINARY);
        if (fd < 0) {
            CloseHandle(hFile);
            throw std::runtime_error("_open_osfhandle failed");
        }

        FILE* file = _fdopen(fd, "w+b");
        if (!file) {
            _close(fd);
            throw std::runtime_error("_fdopen failed");
        }

        fs_ = std::fstream(file);
#else
        path_ = create_unique_path();
        fs_.open(path_, std::ios::in | std::ios::out |
                          std::ios::binary | std::ios::trunc);
        if (!fs_) throw std::runtime_error("Failed to open tmp file");
        unlink(path_.c_str());
#endif
    }

    ~TmpFile() {
        fs_.close();

#ifndef _WIN32
        if (owns_file_ && !path_.empty()) {
            std::remove(path_.c_str());
        }
#endif
    }

    TmpFile(const TmpFile&) = delete;
    TmpFile& operator=(const TmpFile&) = delete;

    TmpFile(TmpFile&& other) noexcept
        : path_(std::move(other.path_)),
          fs_(std::move(other.fs_)),
          owns_file_(other.owns_file_) {
        other.owns_file_ = false;
    }

    TmpFile& operator=(TmpFile&& other) noexcept {
        if (this != &other) {
            fs_.close();
#ifndef _WIN32
            if (owns_file_ && !path_.empty())
                std::remove(path_.c_str());
#endif
            path_ = std::move(other.path_);
            fs_ = std::move(other.fs_);
            owns_file_ = other.owns_file_;
            other.owns_file_ = false;
        }
        return *this;
    }

    std::fstream& stream() { return fs_; }
    const std::filesystem::path& path() const { return path_; }
};

#endif // CHEPP_TMP_FILE_H
