 /** 
 * 日志类头文件, SPD_LOG.h
 **/
#pragma once
#ifndef __SPD_LOG_H__
#define __SPD_LOG_H__

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;

class SPD_LOG
{
public:
    explicit SPD_LOG(const fs::path& log_dir)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!fs::exists(log_dir)) fs::create_directories(log_dir);

        auto console = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file    = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                            (log_dir / make_filename()).string(), true);

        logger_ = std::make_shared<spdlog::logger>("node",
                                                   spdlog::sinks_init_list{console, file});
        spdlog::set_default_logger(logger_);
        spdlog::set_level(spdlog::level::info);
        spdlog::flush_on(spdlog::level::info);
    }

    // 禁止拷贝
    SPD_LOG(const SPD_LOG&)            = delete;
    SPD_LOG& operator=(const SPD_LOG&) = delete;

    // 显式正常析构接口：任何地方都能调用
    static void shutdown()
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (logger_)                // 防止重复调用
        {
            spdlog::info("SPD_LOG::shutdown() 被调用，日志即将 flush …");
            logger_->flush();       // 强制刷盘
            spdlog::drop("node");  // 从全局注册表移除
            logger_.reset();        // 释放 sink，文件句柄关闭
            spdlog::shutdown();     // 可选：清理 spdlog 内部线程
        }
    }

    ~SPD_LOG()
    {
        shutdown();   // 兜底：如果用户忘了调，析构时自动调
    }

private:
    static std::string make_filename()
    {
        auto t  = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &t);
#else
        localtime_r(&t, &tm);
#endif
        char buf[32];
        std::strftime(buf, sizeof(buf), "%F_%H-%M-%S.log", &tm);
        return buf;
    }

    inline static std::shared_ptr<spdlog::logger> logger_;
    inline static std::mutex                      mtx_;
};

#endif