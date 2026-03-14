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

        void
        tune() noexcept {
            if (m_running.test_and_set()) {
                fmt::print(stdout, "error: tuning is not available\n");
                return;
            }
            m_worker = std::jthread([&](const std::stop_token& st) {
                auto res = Arch::make_best_kernels(m_registry, m_layers, st);
                if (st.stop_requested()) {
                    fmt::print(stdout, "info string tuning aborted\n");
                } else {
                    std::scoped_lock lock{m_mtx};
                    m_kernels = res;
                    fmt::print(stdout, "info string tuning successful\n");
                }
                m_running.clear();
            });
        }

        void tune_sync() noexcept {
            auto res = Arch::make_best_kernels(m_registry, m_layers, std::stop_token());
            std::scoped_lock lock{m_mtx};
            m_kernels = res;
        }

        KernelRegistry                 m_registry{};
        Arch::layer_t                  m_layers{};
        Arch::ikernel_t                m_kernels{};
        mutable std::mutex             m_mtx{};
        std::atomic_flag               m_running{};
        std::jthread                   m_worker{};
    };
} // namespace chepp::nnue

#endif