#ifndef CHEPP_NNUE_LAYERS_H_
#define CHEPP_NNUE_LAYERS_H_

#include <algorithm>

#include "utils.h"

#include <any>
#include <coroutine>
#include <cstddef>
#include <functional>
#include <generator>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <variant>

#include <hwy/auto_tune.h>
#include <hwy/nanobenchmark.h>

namespace chepp::nnue {
    namespace layers {
        using namespace chepp::utils;
        using namespace chepp::nnue::utils;

        struct IKernel {
            virtual ~IKernel() noexcept = default;

            [[nodiscard]] virtual std::string
            name() const noexcept = 0;
        };

        struct ILayer {
            virtual ~ILayer() noexcept = default;

            [[nodiscard]] virtual std::string
            name() const noexcept = 0;
        };

        using closure_type = std::function<hwy::FuncOutput(hwy::FuncInput)>;

        template <typename Operation>
        struct ILayerBase : ILayer {
            using ikernel_t = Operation::ikernel_t;

            [[nodiscard]] virtual closure_type
            make_benchmark_closure(const std::shared_ptr<const ikernel_t>&) const noexcept = 0;
        };

        template <typename Operation>
        struct IKernelBase : IKernel, Operation::iexecutable_t, std::enable_shared_from_this<IKernelBase<Operation>> {
            using layer_t = Operation::layer_t;

            using sfinae_flag = void;

            explicit IKernelBase(const std::shared_ptr<layer_t>& layer) noexcept : m_layer{layer} {
                HWY_ASSERT(m_layer);
            }

            [[nodiscard]] double
            benchmark() const noexcept {
                hwy::FuncInput inputs[]{1};
                hwy::Result    results[std::size(inputs)]{};

                hwy::Params params{};
                params.verbose           = false;
                params.target_rel_mad    = 1;
                params.precision_divisor = 1;
                params.seconds_per_eval  = 4e-5;

                auto success = hwy::MeasureClosure(m_layer->make_benchmark_closure(this->shared_from_this()),
                                                   std::data(inputs),
                                                   std::size(inputs),
                                                   std::data(results),
                                                   params);

                if (success != 0) {
                    return static_cast<double>(results->ticks);
                } else {
                    return std::numeric_limits<double>::max();
                }
            }

            [[nodiscard]] const layer_t&
            layer() const noexcept {
                return *m_layer;
            }

          private:
            std::shared_ptr<layer_t> m_layer;
        };

        template <size_t, typename, auto>
        struct Kernel;

        struct default_config_t {
            constexpr bool
            operator==(const default_config_t&) const noexcept {
                return true;
            }
        };

        static constexpr default_config_t default_config;

        struct empty_t {};
        static constexpr empty_t empty;
    } // namespace layers
} // namespace chepp::nnue

template <>
struct std::hash<chepp::nnue::layers::default_config_t> {
    std::size_t
    operator()(const chepp::nnue::layers::default_config_t&) const noexcept {
        return {};
    }
};

namespace chepp::nnue {
    using namespace layers;

    struct KernelRegistry {
        inline static std::mutex&
        type_mtx() noexcept {
            static std::mutex res;
            return res;
        }

        template <typename CfgType>
        inline static std::unordered_map<CfgType, std::type_index>&
        type_mapping() noexcept {
            static std::unordered_map<CfgType, std::type_index> res{};
            return res;
        }

        template <typename Operation>
        using factory_t = std::function<std::shared_ptr<typename Operation::ikernel_t>(
            const std::shared_ptr<typename Operation::layer_t>&)>;

        mutable std::mutex m_mtx;
        std::unordered_map<std::type_index,
                           std::unordered_map<std::size_t, std::unordered_map<std::type_index, std::any>>>
            m_kernels{};

        template <typename CfgType, CfgType cfg>
        static std::type_index
        register_type() {
            std::scoped_lock lock{type_mtx()};
            type_mapping<CfgType>().emplace(cfg, typeid(std::integral_constant<CfgType, cfg>));
            return typeid(std::integral_constant<CfgType, cfg>);
        }

        template <typename CfgType>
        static std::optional<std::type_index>
        get_type_index(CfgType cfg) {
            std::scoped_lock               lock{type_mtx()};
            std::optional<std::type_index> res{};
            if (auto match = type_mapping<CfgType>().find(cfg); match != type_mapping<CfgType>().end()) {
                res.emplace(match->second);
            }
            return res;
        }

        template <typename Operation, std::size_t hwy_target, auto cfg>
        auto
        register_kernel() {
            using layer_t = Operation::layer_t;

            std::scoped_lock lock{m_mtx};

            auto cfg_key = register_type<decltype(cfg), cfg>();
            m_kernels[typeid(Operation)][hwy_target][cfg_key] =
                factory_t<Operation>{[](const std::shared_ptr<layer_t>& layer) {
                    return std::make_shared<typename Kernel<hwy_target, Operation, cfg>::type>(layer);
                }};
        }

        template <typename Operation, typename CfgType>
        auto
        make_kernel(const std::shared_ptr<typename Operation::layer_t>& layer,
                    const std::size_t                                   hwy_target,
                    const CfgType                                       cfg) const {
            using ikernel_t = Operation::ikernel_t;

            std::scoped_lock           lock{m_mtx};
            std::shared_ptr<ikernel_t> kernel{};
            if (auto cfg_key = get_type_index<CfgType>(cfg); cfg_key) {
                kernel = std::any_cast<factory_t<Operation>>(
                    m_kernels.at(typeid(Operation)).at(hwy_target).at(cfg_key.value()))(layer);
            }
            return kernel;
        }

        template <typename Operation>
        auto
        make_all_kernels(const std::shared_ptr<typename Operation::layer_t>& layer) const {
            using ikernel_t = Operation::ikernel_t;

            std::scoped_lock                        lock{m_mtx};
            std::vector<std::shared_ptr<ikernel_t>> res{};
            if (m_kernels.contains(typeid(Operation))) {
                for (const auto& [_, cfgs] : m_kernels.at(typeid(Operation))) {
                    for (const auto& [_1, factory] : cfgs) {
                        res.emplace_back(std::any_cast<factory_t<Operation>>(factory(layer)));
                    }
                }
            }
            return res;
        }

        template <typename Operation>
        auto
        make_best_kernel(const std::shared_ptr<typename Operation::layer_t>& layer,
                         const std::stop_token&                              stop_token) const {
            using ikernel_t = Operation::ikernel_t;

            using result_t = std::shared_ptr<ikernel_t>;

            std::vector<factory_t<Operation>> candidates{};
            for (const auto& [_, cfgs] : m_kernels.at(typeid(Operation))) {
                for (const auto& [_1, factory] : cfgs) {
                    candidates.push_back(std::any_cast<factory_t<Operation>>(factory));
                }
            }

            if (candidates.empty()) {
                return result_t{};
            }

            hwy::AutoTune<size_t> tune{};
            std::vector           indices(candidates.size(), 0uz);
            std::iota(indices.begin(), indices.end(), 0uz);
            tune.SetCandidates(std::move(indices));

            while (!tune.Best()) {
                if (stop_token.stop_requested()) {
                    return result_t{};
                }
                const auto kernel_id = tune.NextConfig();
                tune.NotifyCost(static_cast<uint64_t>(candidates.at(kernel_id)(layer)->benchmark()));
            }
            return candidates.at(*tune.Best())(layer);
        }
    };
} // namespace chepp::nnue

#endif
