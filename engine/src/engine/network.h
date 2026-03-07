#ifndef SIMPLE_NNUE_H
#define SIMPLE_NNUE_H

#include "core.h"
#include "nnue.h"

#include <atomic>

namespace chepp::nnue {
    struct NetworkHandle {
        NetworkHandle() noexcept : m_layers(Arch::make_layers()) {
            register_all_layers(m_registry);
            m_network = std::make_shared<Arch::Network>(Arch::make_kernels(m_registry, m_layers));
        }

        auto
        get() const noexcept {
            std::scoped_lock lock{m_mtx};
            return m_network;
        }

        void
        tune() noexcept {
            if (m_running.test_and_set()) {
                fmt::print(std::cerr, "error: tuning is not available\n");
                return;
            }
            m_worker = std::jthread([&](const std::stop_token& st) {
                auto res = Arch::make_best_kernels(m_registry, m_layers, st);
                if (st.stop_requested()) {
                    fmt::print(std::cerr, "info string tuning aborted\n");
                } else {
                    std::scoped_lock lock{m_mtx};
                    m_network = std::make_shared<Arch::Network>(res);
                    fmt::print(std::cout, "info string tuning successful\n");
                }
                m_running.clear();
            });
        }

        KernelRegistry                 m_registry{};
        Arch::layer_t                  m_layers{};
        std::shared_ptr<Arch::Network> m_network{};
        mutable std::mutex             m_mtx{};
        std::atomic_flag               m_running{};
        std::jthread                   m_worker{};
    };
} // namespace chepp::nnue

#endif