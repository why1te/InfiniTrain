#pragma once
// Optional diagnostics for one training thread; never used to choose a layout.
#include "infini_train/include/core/runtime/device_guard.h"
#include "infini_train/include/nn/modules/module.h"
#include "infini_train/include/nn/parallel/pp/pipeline_layout.h"
#include "infini_train/include/tensor.h"
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#ifdef USE_CUDA
#include <nvtx3/nvToolsExt.h>
#endif

namespace infini_train::utils {
inline std::string ToDiagnosticParameterName(std::string_view name,
                                             const std::shared_ptr<const nn::parallel::PipelineLayout> &layout,
                                             int pp_rank) {
    constexpr std::string_view prefix = "transformer.h.";
    if (!layout || !name.starts_with(prefix)) {
        return std::string(name);
    }
    const auto &stage = layout->GetStage(pp_rank);
    const auto tail = name.substr(prefix.size());
    const auto dot = tail.find('.');
    const auto token = tail.substr(0, dot);
    nn::parallel::LayerIndex local = -1;
    const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), local);
    if (token.empty() || error != std::errc{} || end != token.data() + token.size() || local < 0) {
        throw std::out_of_range("invalid local layer in precision name: " + std::string(name));
    }
    for (const auto &chunk : stage.chunks) {
        if (local < chunk.layer_range.Size()) {
            return std::string(prefix) + std::to_string(chunk.layer_range.begin + local)
                 + (dot == std::string_view::npos ? "" : std::string(tail.substr(dot)));
        }
        local -= chunk.layer_range.Size();
    }
    throw std::out_of_range("local layer is not owned by stage: " + std::string(name));
}

class PipelineDiagnostics {
public:
    using NameMapper = std::function<std::string(const std::string &)>;
    // Empty paths disable each output. rank is PP-group-local, not CUDA index.
    PipelineDiagnostics(std::shared_ptr<nn::Module> model, NameMapper mapper, Device device, int rank, bool owns_loss,
                        const std::string &gradient_root, const std::string &timing_root)
        : model_(std::move(model)), mapper_(std::move(mapper)), device_(device), rank_(rank), owns_loss_(owns_loss) {
        if (!gradient_root.empty()) {
            gradients_ = NewRankDirectory(gradient_root, rank);
        }
        if (!timing_root.empty()) {
            const auto path = NewRankDirectory(timing_root, rank);
            tasks_.exceptions(std::ios::failbit | std::ios::badbit);
            stages_.exceptions(std::ios::failbit | std::ios::badbit);
            tasks_.open(path / "tasks.csv");
            stages_.open(path / "stage-times.csv");
            tasks_ << "step,stage,microbatch,gid,direction,start_ms,end_ms\n";
            stages_ << "step,stage,forward_ms,backward_ms\n";
            tasks_ << std::setprecision(17);
            stages_ << std::setprecision(17);
        }
    }
    bool TimingEnabled() const { return stages_.is_open(); }
    void Synchronize() const { core::GetDeviceGuardImpl(device_.type())->SynchronizeDevice(device_); }
    // step is the zero-based optimizer iteration, not a scheduler task slot.
    void BeginStep(int step) {
        if (step < 0 || step <= step_) {
            throw std::invalid_argument("non-increasing diagnostic step");
        }
        step_ = step;
        forward_ms_ = backward_ms_ = 0;
        task_count_ = 0;
        epoch_ = Clock::now();
    }
    double NowMs() const { return std::chrono::duration<double, std::milli>(Clock::now() - epoch_).count(); }
    void RecordTask(int mb, int gid, bool forward, double begin, double end) {
        if (step_ < 0 || mb < 0 || gid < 0 || !std::isfinite(end) || end < begin) {
            throw std::runtime_error("invalid task timing");
        }
        tasks_ << step_ << ',' << rank_ << ',' << mb << ',' << gid << ',' << (forward ? "forward" : "backward") << ','
               << begin << ',' << end << '\n';
        (forward ? forward_ms_ : backward_ms_) += end - begin;
        ++task_count_;
    }
    // Capture the first full PP batch before splitting it into microbatches.
    // Keep original dtype/shape/bytes: SaveAsNpy only supports float32 tensors.
    void RecordBatch(const Tensor &input, const Tensor &target) {
        if (gradients_.empty() || dumped_ || batch_saved_) {
            return;
        }
        if (step_ < 0) {
            throw std::runtime_error("diagnostics not initialized");
        }
        auto save = [&](const Tensor &tensor, const char *name) {
            auto cpu = Tensor(tensor).To(Device());
            std::ofstream output;
            output.exceptions(std::ios::failbit | std::ios::badbit);
            output.open(gradients_ / name, std::ios::binary);
            output << static_cast<int>(cpu.Dtype()) << '\n';
            for (auto dim : cpu.Dims()) { output << dim << ','; }
            output << '\n';
            output.write(static_cast<const char *>(cpu.DataPtr()), cpu.SizeInBytes());
            output.close();
        };
        save(input, "inputs.bin");
        save(target, "targets.bin");
        batch_saved_ = true;
    }
    void CaptureError() noexcept { error_ = std::current_exception(); }
    // Called once, immediately before the first optimizer update on each rank.
    void BeforeOptimizerStep() {
        if (gradients_.empty() || dumped_) {
            return;
        }
        if (step_ < 0 || !model_) {
            throw std::runtime_error("diagnostics not initialized");
        }
        std::ofstream manifest;
        manifest.exceptions(std::ios::failbit | std::ios::badbit);
        manifest.open(gradients_ / "parameters.tsv");
        manifest << "name\tshape\tstatus\n";
        std::unordered_set<const Tensor *> seen;
        std::set<std::string> names;
        bool missing = false;
        for (const auto &[local, param] : model_->NamedParameters("", true, false)) {
            if (local.starts_with("__pp") || !param || !param->requires_grad()) {
                continue;
            }
            if (!seen.insert(param.get()).second) {
                continue;
            }
            const auto name = mapper_(local);
            if (name.empty()
                || name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
                       != std::string::npos
                || !names.insert(name).second) {
                throw std::runtime_error("invalid/duplicate global parameter name: " + name);
            }
            manifest << name << '\t';
            for (auto dimension : param->Dims()) { manifest << dimension << ','; }
            const auto gradient = param->grad();
            if (!gradient) {
                manifest << "\tMISSING\n";
                missing = true;
                continue;
            }
            if (gradient->Dims() != param->Dims()) {
                throw std::runtime_error("gradient shape mismatch: " + name);
            }
            auto cpu = gradient->To(Device()).To(DataType::kFLOAT32);
            const auto *values = static_cast<const float *>(cpu.DataPtr());
            for (std::size_t i = 0; i < cpu.NumElements(); ++i) {
                if (!std::isfinite(values[i])) {
                    throw std::runtime_error("non-finite gradient: " + name);
                }
            }
            cpu.SaveAsNpy((gradients_ / (name + ".npy")).string());
            manifest << "\tOK\n";
        }
        manifest.close();
        if (names.empty() || missing) {
            throw std::runtime_error("empty/incomplete parameter gradients");
        }
        dumped_ = true;
    }
    void EndStep(float loss) {
        if (error_) {
            std::rethrow_exception(error_);
        }
        if (TimingEnabled()) {
            if (task_count_ == 0) {
                throw std::runtime_error("no pipeline tasks recorded");
            }
            stages_ << step_ << ',' << rank_ << ',' << forward_ms_ << ',' << backward_ms_ << '\n';
            stages_.flush();
            tasks_.flush();
        }
        if (dumped_ && !loss_saved_ && owns_loss_) {
            if (!std::isfinite(loss)) {
                throw std::runtime_error("non-finite loss");
            }
            std::ofstream output;
            output.exceptions(std::ios::failbit | std::ios::badbit);
            output.open(gradients_ / "loss.txt");
            output << std::setprecision(9) << loss << '\n';
            output.close();
            loss_saved_ = true;
        }
    }

private:
    static std::filesystem::path NewRankDirectory(const std::string &root, int rank) {
        if (rank < 0) {
            throw std::invalid_argument("negative PP rank");
        }
        std::filesystem::create_directories(root);
        auto path = std::filesystem::path(root) / ("rank-" + std::to_string(rank));
        if (!std::filesystem::create_directory(path)) {
            throw std::runtime_error("refusing overwrite: " + path.string());
        }
        return path;
    }
    using Clock = std::chrono::steady_clock;
    std::shared_ptr<nn::Module> model_;
    NameMapper mapper_;
    Device device_;
    int rank_, step_ = -1, task_count_ = 0;
    bool owns_loss_, dumped_ = false, loss_saved_ = false, batch_saved_ = false;
    std::filesystem::path gradients_;
    std::ofstream tasks_, stages_;
    Clock::time_point epoch_;
    double forward_ms_ = 0, backward_ms_ = 0;
    std::exception_ptr error_;
};
// Explicitly scoped per training thread; restored on every exit path.
inline thread_local PipelineDiagnostics *active_pipeline_diagnostics = nullptr;
class PipelineDiagnosticsScope {
public:
    explicit PipelineDiagnosticsScope(PipelineDiagnostics *current) : previous_(active_pipeline_diagnostics) {
        active_pipeline_diagnostics = current;
    }
    ~PipelineDiagnosticsScope() { active_pipeline_diagnostics = previous_; }
    PipelineDiagnosticsScope(const PipelineDiagnosticsScope &) = delete;
    PipelineDiagnosticsScope &operator=(const PipelineDiagnosticsScope &) = delete;

private:
    PipelineDiagnostics *previous_;
};
inline void DumpPipelineParameterGradients() {
    if (active_pipeline_diagnostics) {
        active_pipeline_diagnostics->BeforeOptimizerStep();
    }
}
#ifdef USE_CUDA
inline thread_local int active_pipeline_trace_step = -1;
#endif
inline int CurrentPipelineTraceStep() {
#ifdef USE_CUDA
    return active_pipeline_trace_step;
#else
    return -1;
#endif
}
inline std::string BuildPipelineP2PTraceLabel(int step, int mb, int boundary, int peer, std::size_t bytes,
                                              std::size_t tensors, bool forward, bool send) {
    if (step < 0 || mb < 0 || boundary < 0 || peer < 0 || bytes == 0 || tensors == 0) {
        throw std::invalid_argument("invalid pipeline P2P trace metadata");
    }
    const auto direction = forward ? "forward" : "backward";
    return "pipeline p2p step=" + std::to_string(step) + " mb=" + std::to_string(mb)
         + " boundary=" + std::to_string(boundary) + " direction=" + direction + " role=" + (send ? "send" : "recv")
         + " peer=" + std::to_string(peer) + " bytes=" + std::to_string(bytes) + " tensors=" + std::to_string(tensors)
         + " message=" + std::to_string(step) + ':' + std::to_string(mb) + ':' + std::to_string(boundary) + ':'
         + direction;
}
class PipelineP2PTrace {
public:
    PipelineP2PTrace(int step, int mb, int boundary, int peer, std::size_t bytes, std::size_t tensors, bool forward,
                     bool send) {
#ifdef USE_CUDA
        if (step >= 0) {
            const auto label = BuildPipelineP2PTraceLabel(step, mb, boundary, peer, bytes, tensors, forward, send);
            nvtxRangePushA(label.c_str());
            enabled_ = true;
        }
#else
        (void)step;
        (void)mb;
        (void)boundary;
        (void)peer;
        (void)bytes;
        (void)tensors;
        (void)forward;
        (void)send;
#endif
    }
    ~PipelineP2PTrace() {
#ifdef USE_CUDA
        if (enabled_) {
            nvtxRangePop();
        }
#endif
    }
    PipelineP2PTrace(const PipelineP2PTrace &) = delete;
    PipelineP2PTrace &operator=(const PipelineP2PTrace &) = delete;

private:
#ifdef USE_CUDA
    bool enabled_ = false;
#endif
};
// An optional outer range gives Nsight a training-step label for nested task ranges.
class PipelineTaskTraceStep {
public:
    PipelineTaskTraceStep(int step, bool enabled) {
#ifdef USE_CUDA
        if (enabled) {
            previous_ = active_pipeline_trace_step;
            active_pipeline_trace_step = step;
            const auto label = "pipeline step=" + std::to_string(step);
            nvtxRangePushA(label.c_str());
            enabled_ = true;
        }
#else
        (void)step;
        (void)enabled;
#endif
    }
    ~PipelineTaskTraceStep() {
#ifdef USE_CUDA
        if (enabled_) {
            nvtxRangePop();
            active_pipeline_trace_step = previous_;
        }
#endif
    }
    PipelineTaskTraceStep(const PipelineTaskTraceStep &) = delete;
    PipelineTaskTraceStep &operator=(const PipelineTaskTraceStep &) = delete;

private:
#ifdef USE_CUDA
    int previous_ = -1;
    bool enabled_ = false;
#endif
};
class PipelineTaskTimer {
public:
    PipelineTaskTimer(int mb, int gid, int owner, bool forward)
        : diagnostic_(active_pipeline_diagnostics), mb_(mb), gid_(gid), forward_(forward) {
        if (diagnostic_ && diagnostic_->TimingEnabled()) {
            diagnostic_->Synchronize();
            begin_ = diagnostic_->NowMs();
        } else {
            diagnostic_ = nullptr;
        }
#ifdef USE_CUDA
        if (active_pipeline_trace_step >= 0) {
            const auto label = "pipeline task step=" + std::to_string(active_pipeline_trace_step)
                             + " mb=" + std::to_string(mb) + " gid=" + std::to_string(gid)
                             + " owner=" + std::to_string(owner) + (forward ? " forward" : " backward");
            nvtxRangePushA(label.c_str());
            traced_ = true;
        }
#else
        (void)owner;
#endif
    }
    ~PipelineTaskTimer() noexcept {
        if (diagnostic_) {
            try {
                diagnostic_->Synchronize();
                diagnostic_->RecordTask(mb_, gid_, forward_, begin_, diagnostic_->NowMs());
            } catch (...) { diagnostic_->CaptureError(); }
        }
#ifdef USE_CUDA
        if (traced_) {
            nvtxRangePop();
        }
#endif
    }
    PipelineTaskTimer(const PipelineTaskTimer &) = delete;
    PipelineTaskTimer &operator=(const PipelineTaskTimer &) = delete;

private:
    PipelineDiagnostics *diagnostic_;
    int mb_, gid_;
    bool forward_;
    double begin_ = 0;
#ifdef USE_CUDA
    bool traced_ = false;
#endif
};
} // namespace infini_train::utils
