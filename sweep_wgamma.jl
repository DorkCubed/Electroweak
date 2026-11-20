using Flux
using Plots, ProgressMeter
using CUDA
using CSV, DataFrames, Random, Statistics, StatsBase

const dev = CUDA.functional() ? Flux.gpu_device() : Flux.cpu_device()
const cpu_dev = Flux.cpu_device()

# 1. Load data and setup bins
df = DataFrame(CSV.File(joinpath(@__DIR__, "wgamma_master_dataset.csv")))
scale = 294.0 / sum(df.w_SM)

bins_phi = range(-Float64(π), Float64(π), length=21)
bin_w_phi = step(bins_phi)
w_sm_phi = (df.w_SM ./ 20.0) .* (scale / bin_w_phi)
w_int_phi = df.w_int .* (scale / bin_w_phi)

bins_nn = range(-1.0, 1.0, length=21)
bin_w_nn = step(bins_nn)
w_sm_nn = (df.w_SM ./ 20.0) .* (scale / bin_w_nn)
w_int_nn = df.w_int .* (scale / bin_w_nn)

# 2. Features and CP Transformation
feat_cols = [:lep_pt, :lep_eta, :lep_phi, :lep_charge, :lep_flavour,
    :pho_pt, :pho_eta, :pho_phi, :met, :met_phi]

wrap_pi(θ) = mod(θ + Float32(π), 2Float32(π)) - Float32(π)

function cp_transform(x)
    return Float32[
        x[1], -x[2], wrap_pi(x[3] + Float32(π)), -x[4], x[5],
        x[6], -x[7], wrap_pi(x[8] + Float32(π)), x[9], wrap_pi(x[10] + Float32(π))
    ]
end

# 3. Precompute 5-Fold Augmented Data (avoids rebuilding DataFrames on every sweep trial)
Random.seed!(67)
n_events = nrow(df)
folds = shuffle(1:n_events) .% 5 .+ 1

struct FoldData
    X_train_norm::Matrix{Float32}
    Y_train::Flux.OneHotArray
    W_train::Vector{Float32}
    X_test_norm::Matrix{Float32}
    test_mask::BitVector
end

println("Precomputing 5-fold CP-augmented datasets...")
fold_datasets = Vector{FoldData}(undef, 5)

for k in 1:5
    train_mask = folds .!= k
    test_mask = folds .== k

    train_df = df[train_mask, :]
    test_df = df[test_mask, :]

    W_sm = sum(train_df.w_SM)
    W_pos = sum(abs, train_df.w_int[train_df.w_int.>0])
    W_neg = sum(abs, train_df.w_int[train_df.w_int.<0])

    n_train = nrow(train_df)
    X_train = Matrix{Float32}(undef, 10, 4 * n_train)
    y_train = Vector{Int}(undef, 4 * n_train)
    W_train = Vector{Float32}(undef, 4 * n_train)

    idx = 1
    for row in eachrow(train_df)
        x = Float32[row[col] for col in feat_cols]
        xcp = cp_transform(x)
        a = Float32(row.w_SM / W_sm)
        wint = Float32(row.w_int)
        b = abs(wint) / Float32(wint > 0 ? W_pos : W_neg)
        t_orig = wint > 0 ? 2 : 3
        t_cp = wint > 0 ? 3 : 2

        X_train[:, idx] = x
        y_train[idx] = 1
        W_train[idx] = a
        X_train[:, idx+1] = x
        y_train[idx+1] = t_orig
        W_train[idx+1] = b
        X_train[:, idx+2] = xcp
        y_train[idx+2] = 1
        W_train[idx+2] = a
        X_train[:, idx+3] = xcp
        y_train[idx+3] = t_cp
        W_train[idx+3] = b
        idx += 4
    end

    Y_train = Flux.onehotbatch(y_train, 1:3)

    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2)
    σ[σ.==0] .= 1.0f0

    X_train_norm = (X_train .- μ) ./ σ
    X_test = reduce(hcat, [Float32[row[col] for col in feat_cols] for row in eachrow(test_df)])
    X_test_norm = (X_test .- μ) ./ σ

    fold_datasets[k] = FoldData(X_train_norm, Y_train, W_train, X_test_norm, test_mask)
end

# 4. Loss & Target Evaluation Functions
loss_fn(m, x, y, w) = sum(w .* vec(Flux.Losses.logitcrossentropy(m(x), y; agg=identity))) / sum(w)

"""
Evaluates:
1. `green_height`: Peak height of the SM/20 histogram [fb].
2. `red_range`: Horizontal spread of the interference curve across [-1, 1],
   combining active bin span and the 2%–98% |w_int|-weighted quantile range.
3. `target`: Composite score `green_height * (red_range^2)` that rewards both a tall
   green peak and a wide horizontal spread for the red curve.
"""
function evaluate_target(O_NN::Vector{Float32}, w_sm_nn, w_int_nn, bins_nn)
    o_clamped = clamp.(Float64.(O_NN), first(bins_nn) + 1e-6, last(bins_nn) - 1e-6)

    h_green = fit(Histogram, o_clamped, weights(Float64.(w_sm_nn)), bins_nn).weights
    h_red = fit(Histogram, o_clamped, weights(Float64.(w_int_nn)), bins_nn).weights

    green_height = maximum(h_green)

    # Horizontal span of bins where red histogram is non-negligible (> 1% of peak)
    red_peak = maximum(abs, h_red)
    active_bins = findall(x -> abs(x) >= max(0.02 * red_peak, 1e-3), h_red)
    bin_span = isempty(active_bins) ? 0.0 : (maximum(active_bins) - minimum(active_bins) + 1) * step(bins_nn)

    # Smooth 2%-98% weighted quantile span of O_NN weighted by |w_int|
    w_abs_int = weights(abs.(Float64.(w_int_nn)))
    q_span = quantile(o_clamped, w_abs_int, 0.98) - quantile(o_clamped, w_abs_int, 0.02)

    red_range = 0.5 * (bin_span + q_span)
    target = green_height * (red_range^2)

    return (target=target, green_height=green_height, red_range=red_range)
end

# 5. Single 5-Fold CV Trial Runner
function run_cv_trial(; neurons::Int, lr::Float64, epochs::Int, batchsize::Int)
    Random.seed!(67) # Fixed seed for fair comparison across hyperparameter trials

    P_SM = zeros(Float32, n_events)
    P_plus = zeros(Float32, n_events)
    P_minus = zeros(Float32, n_events)
    O_NN = zeros(Float32, n_events)

    p = Progress(5 * epochs; desc="Trial (n=$neurons, lr=$lr, ep=$epochs, bs=$batchsize): ")

    for k in 1:5
        fd = fold_datasets[k]

        model = Chain(
            Dense(10 => neurons, relu),
            Dense(neurons => 2 * neurons, relu),
            Dense(2 * neurons => neurons, relu),
            Dense(neurons => 3)
        ) |> dev

        opt = Flux.setup(Adam(lr), model)

        n_aug = size(fd.X_train_norm, 2)
        sub_n = div(n_aug, 2)

        for _ in 1:epochs
            idx = randperm(n_aug)[1:sub_n]
            loader = Flux.DataLoader(
                (fd.X_train_norm[:, idx], fd.Y_train[:, idx], fd.W_train[idx]),
                batchsize=batchsize, shuffle=true
            ) |> dev
            Flux.train!(loss_fn, model, loader, opt)
            next!(p)
        end

        probs = softmax(model(fd.X_test_norm |> dev)) |> cpu_dev
        view(P_SM, fd.test_mask) .= probs[1, :]
        view(P_plus, fd.test_mask) .= probs[2, :]
        view(P_minus, fd.test_mask) .= probs[3, :]
        view(O_NN, fd.test_mask) .= probs[2, :] .- probs[3, :]
    end

    metrics = evaluate_target(O_NN, w_sm_nn, w_int_nn, bins_nn)
    return (
        metrics=metrics,
        P_SM=P_SM, P_plus=P_plus, P_minus=P_minus, O_NN=O_NN
    )
end

# 6. Sequential Greedy Coordinate Sweep (Keep accepting until target decreases)
best_config = Dict{Symbol,Any}(
    :lr => 8e-3,
    :neurons => 128,
    :epochs => 150,
    :batchsize => 64
)

println("\n--- Evaluating Baseline Configuration ---")
best_res = run_cv_trial(;
    neurons=best_config[:neurons],
    lr=best_config[:lr],
    epochs=best_config[:epochs],
    batchsize=best_config[:batchsize]
)
println("Baseline -> Target: $(round(best_res.metrics.target, digits=3)) | " *
        "Green Height: $(round(best_res.metrics.green_height, digits=2)) fb | " *
        "Red Range: $(round(best_res.metrics.red_range, digits=3))")

"""
Sweeps `param_name` through `candidates` starting from the current best value.
Steps in the improving direction and keeps accepting new values until the target decreases.
"""
function sweep_parameter!(param_name::Symbol, candidates::Vector, best_config, best_res)
    println("\n====================================================")
    println("Sweeping $(param_name): candidates =$(candidates)")
    println("Starting $(param_name) = $(best_config[param_name]) (Target =$(round(best_res.metrics.target, digits=3)))")
    println("====================================================")

    curr_val = best_config[param_name]
    start_idx = findfirst(==(curr_val), candidates)

    # Build ordered directions away from current value (forward first, then backward if needed)
    directions = Vector{Vector{eltype(candidates)}}()
    if start_idx === nothing
        push!(directions, candidates)
    else
        if start_idx < length(candidates)
            push!(directions, candidates[(start_idx+1):end])
        end
        if start_idx > 1
            push!(directions, reverse(candidates[1:(start_idx-1)]))
        end
    end

    improved_in_direction = false
    for dir_candidates in directions
        for val in dir_candidates
            trial_config = copy(best_config)
            trial_config[param_name] = val

            res = run_cv_trial(;
                neurons=trial_config[:neurons],
                lr=trial_config[:lr],
                epochs=trial_config[:epochs],
                batchsize=trial_config[:batchsize]
            )

            println("Tested $(param_name) = $(val) -> Target:$(round(res.metrics.target, digits=3)) " *
                    "(Green Height: $(round(res.metrics.green_height, digits=2)) fb, " *
                    "Red Range: $(round(res.metrics.red_range, digits=3)))")

            if res.metrics.target > best_res.metrics.target
                println("  [ACCEPTED] Target improved from $(round(best_res.metrics.target, digits=3)) to$(round(res.metrics.target, digits=3))")
                best_config[param_name] = val
                best_res = res
                improved_in_direction = true
            else
                println("  [STOP] Target decreased ($(round(res.metrics.target, digits=3)) <= $(round(best_res.metrics.target, digits=3))). Stopping sweep for $(param_name).")
                break
            end
        end
        # If we already found improvements in the first direction, don't reverse direction
        improved_in_direction && break
    end

    println("Best $(param_name) locked at:$(best_config[param_name])")
    return best_res
end

# Ordered sweep schedule (one hyperparameter at a time)
sweep_schedule = [
    (:lr, [8e-3, 3e-3, 1e-3, 5e-4, 2e-4]),
    (:neurons, [64, 128, 256, 512]),
    (:epochs, [100, 150, 200, 300]),
    (:batchsize, [64, 128, 256, 512])
]

for (param, candidates) in sweep_schedule
    global best_res = sweep_parameter!(param, candidates, best_config, best_res)
end

println("\n================ FINAL BEST HYPERPARAMETERS ================")
println("  Learning Rate : $(best_config[:lr])")
println("  Neurons       : $(best_config[:neurons])")
println("  Epochs        : $(best_config[:epochs])")
println("  Batch Size    : $(best_config[:batchsize])")
println("  Final Target  : $(round(best_res.metrics.target, digits=3))")
println("  Green Height  : $(round(best_res.metrics.green_height, digits=2)) fb")
println("  Red Range     : $(round(best_res.metrics.red_range, digits=3))")
println("============================================================")

# 7. Figure 4: Differential cross section vs O_NN (using best model predictions)
O_NN = best_res.O_NN
P_SM = best_res.P_SM
P_plus = best_res.P_plus
P_minus = best_res.P_minus

p4 = stephist(O_NN, weights=weights(w_sm_nn), bins=bins_nn,
    label="SM/20 (c_ΦWB model)", color=:green, lw=1.5,
    xlabel="O_NN", ylabel="dσ/dO_NN [fb]",
    title="inclusive Wγ production (lr=$(best_config[:lr]), n=$(best_config[:neurons]), ep=$(best_config[:epochs]))",
    legend=:topright)
stephist!(p4, O_NN, weights=weights(w_int_nn), bins=bins_nn,
    label="c_ΦWB/Λ² = 1 TeV⁻²", color=:red, lw=1.5)
savefig(p4, joinpath(@__DIR__, "Figure_4.png"))

# 8. Save output dataset from the best configuration
df.cv_fold = folds
df.P_SM = P_SM
df.P_plus = P_plus
df.P_minus = P_minus
df.O_NN = O_NN
CSV.write(joinpath(@__DIR__, "wgamma_predictions.csv"), df)