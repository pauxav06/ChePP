#ifndef SIMPLE_NNUE_H
#define SIMPLE_NNUE_H

#include "core.h"
#include "nnue.h"

#include <atomic>

namespace chepp::nnue {
    struct NetworkHandle {
        NetworkHandle() noexcept : m_layers(Arch::make_layers()) {
            register_all_layers(m_registry);
            m_kernels = Arch::make_kernels(m_registry, m_layers);
        }

        auto
        get() const noexcept {
            std::scoped_lock lock{m_mtx};
            return std::make_shared<Arch::Network>(m_kernels); //TODO make unique
        }

        bool
        tune() noexcept {
            const std::unique_lock lock(m_mtx, std::try_to_lock);
            if (!lock.owns_lock()) {
                fmt::print(stderr, "error: tuning is not available\n");
                return false;
            }
            m_worker = std::jthread([&](const std::stop_token& st) {
                const auto res = Arch::make_best_kernels(m_registry, m_layers, st);
                if (st.stop_requested()) {
                    fmt::print(stdout, "info string tuning aborted\n");
                    return;
                }
                m_kernels = res;
                fmt::print(stdout, "info string tuning successful\n");
            });
            return true;
        }

        bool tune_sync() noexcept {
            const std::unique_lock lock(m_mtx, std::try_to_lock);
            if (!lock.owns_lock()) {
                fmt::print(stderr, "error: tuning is not available\n");
                return false;
            }
            const auto res = Arch::make_best_kernels(m_registry, m_layers, std::stop_token());
            m_kernels = res;
            return true;
        }

        KernelRegistry                 m_registry{};
        Arch::layer_t                  m_layers{};
        Arch::ikernel_t                m_kernels{};
        mutable std::mutex             m_mtx{};
        std::jthread                   m_worker{};
    };
} // namespace chepp::nnue

#endif