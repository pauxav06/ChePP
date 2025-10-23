#ifndef CHEPP_LAYER_CACHE_H_
#define CHEPP_LAYER_CACHE_H_

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>
#include <unordered_map>

class GlobalCache {
  public:
    using key_type = uint64_t;

    template <typename T>
    static std::shared_ptr<T> get(const key_type& key) {
        auto&            map = get_map<T>();
        std::scoped_lock lock(map.mutex);
        if (auto it = map.cache.find(key); it != map.cache.end()) {
            if (auto ptr = it->second.lock()) return std::static_pointer_cast<T>(ptr);
        }
        return nullptr;
    }

    template <typename T>
    static void put(const key_type& key, const std::shared_ptr<T>& value) {
        auto&            map = get_map<T>();
        std::scoped_lock lock(map.mutex);
        map.cache[key] = value;
    }

    template <typename T>
    static std::shared_ptr<T> get_or_create(const key_type& key, std::function<std::shared_ptr<T>()> factory) {
        if (auto existing = get<T>(key)) return existing;

        auto value = factory();
        put<T>(key, value);
        return value;
    }

    template <typename T>
    static void clear() {
        auto&            map = get_map<T>();
        std::scoped_lock lock(map.mutex);
        map.cache.clear();
    }

  private:
    struct TypeMap {
        std::mutex                                        mutex;
        std::unordered_map<key_type, std::weak_ptr<void>> cache;
    };

    template <typename T>
    static TypeMap& get_map() {
        static TypeMap map;
        return map;
    }
};

inline GlobalCache g_cache{};

#endif
