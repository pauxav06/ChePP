#ifndef CHEPP_NNUE_LAYERS_H_
#define CHEPP_NNUE_LAYERS_H_

#include <algorithm>

#include "utils.h"

#include <coroutine>
#include <cstddef>
#include <functional>
#include <generator>
#include <memory>
#include <memory_resource>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <variant>

#include <hwy/auto_tune.h>
#include <hwy/nanobenchmark.h>
#include <hwy/targets.h>

namespace chepp::nnue {
    namespace layers {

        using namespace utils;

        using extent_type = size_t;

        template <size_t, typename, auto>
        struct Kernel;

        struct KernelBase {
            using sfinae_flag     = void;
            virtual ~KernelBase() = default;
        };

        struct ILayer {
            virtual ~ILayer() = default;
            [[nodiscard]] virtual double
            benchmark(const KernelBase&) const = 0;
        };

        struct KernelKey {
            std::size_t     target;
            std::type_index cfg;

            bool
            operator==(const KernelKey&) const = default;
        };
    } // namespace layers
} // namespace chepp::nnue

template <>
struct std::hash<chepp::nnue::layers::KernelKey> {
    std::size_t
    operator()(const chepp::nnue::layers::KernelKey& k) const noexcept {
        return std::hash<std::type_index>{}(k.cfg) ^ std::hash<std::size_t>{}(k.target);
    }
};

namespace chepp::nnue {
    using namespace layers;
    struct KernelRegistry {
        template <typename CfgType>
        inline static std::unordered_map<CfgType, std::type_index> cfg_mapping{};

        template <typename CfgType>
        inline static std::mutex cfg_mtx;

        std::mutex m_kernels_mtx;

        using KernelFactory = std::function<std::shared_ptr<KernelBase>(std::shared_ptr<ILayer>)>;
        std::unordered_map<std::type_index, std::unordered_map<KernelKey, KernelFactory>> m_kernels{};

        template <typename CfgType, CfgType cfg>
        static std::type_index
        register_cfg_key() {
            std::lock_guard lock{cfg_mtx<CfgType>};
            cfg_mapping<CfgType>.emplace(cfg, typeid(std::integral_constant<CfgType, cfg>));
            return typeid(std::integral_constant<CfgType, cfg>);
        }

        template <typename CfgType>
        static std::optional<std::type_index>
        get_cfg_key(CfgType cfg) {
            std::lock_guard                lock{cfg_mtx<CfgType>};
            std::optional<std::type_index> res{};
            if (auto match = cfg_mapping<CfgType>.find(cfg); match != cfg_mapping<CfgType>.end()) {
                res.emplace(match->second);
            }
            return res;
        }

        template <typename Layer, std::size_t hwy_target, auto cfg>
            requires std::is_base_of_v<ILayer, Layer>
        auto
        register_kernel() {
            using layer_t = std::decay_t<Layer>;
            std::lock_guard lock{m_kernels_mtx};
            auto            cfg_key = register_cfg_key<decltype(cfg), cfg>();
            m_kernels[typeid(layer_t)][KernelKey{hwy_target, cfg_key}] =
                [](const std::shared_ptr<ILayer>& layer) -> std::shared_ptr<KernelBase> {
                auto typed = std::dynamic_pointer_cast<layer_t>(layer);
                return std::make_shared<typename Kernel<hwy_target, Layer, cfg>::type>(typed);
            };
            return KernelKey{hwy_target, cfg_key};
        }

        template <typename Layer, typename CfgType>
            requires std::is_base_of_v<ILayer, Layer>
        std::optional<std::shared_ptr<typename Layer::IKernel>>
        make_kernel(const std::shared_ptr<Layer>& layer, const std::size_t hwy_target, const CfgType cfg) {
            using layer_t  = std::decay_t<Layer>;
            using kernel_t = layer_t::IKernel;
            std::lock_guard lock{m_kernels_mtx};
            return get_cfg_key<CfgType>(cfg).transform([&](const std::type_index cfg_key) {
                return std::dynamic_pointer_cast<kernel_t>(
                    m_kernels.at(typeid(Layer)).at(KernelKey{hwy_target, cfg_key})(layer));
            });
        }

        template <typename Layer>
            requires std::is_base_of_v<ILayer, Layer>
        auto
        make_all_kernels(const std::shared_ptr<Layer>& layer) {
            using layer_t  = std::decay_t<Layer>;
            using kernel_t = layer_t::IKernel;

            std::lock_guard                        lock{m_kernels_mtx};
            std::vector<std::shared_ptr<kernel_t>> res{};

            if (m_kernels.contains(typeid(Layer))) {
                for (const auto& [_, factory] : m_kernels.at(typeid(Layer))) {
                    res.emplace_back(std::dynamic_pointer_cast<kernel_t>(factory(layer)));
                }
            }

            return res;
        }

        template <typename Layer>
            requires std::is_base_of_v<ILayer, Layer>
        auto
        get_best_kernel(const std::shared_ptr<Layer>& layer) {
            auto candidates = make_all_kernels(layer);

            hwy::AutoTune<int, 12> tune{};
            std::vector            indices(candidates.size(), 0);
            std::iota(indices.begin(), indices.end(), 0);
            tune.SetCandidates(std::move(indices));

            while (!tune.Best()) {
                auto kernel_id = tune.NextConfig();
                tune.NotifyCost(layer->benchmark(*candidates.at(kernel_id)));
            }
            return candidates.at(*tune.Best());
        }
    };
} // namespace chepp::nnue

#endif
