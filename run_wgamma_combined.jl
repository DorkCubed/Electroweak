using Flux
using Plots, ProgressMeter
using CUDA
using CSV, DataFrames, Random, Statistics, StatsBase, Printf

const dev = CUDA.functional() ? Flux.gpu_device() : Flux.cpu_device()
const cpu_dev = Flux.cpu_device()

println("="^60)
println("Wγ 5-Fold Cross Validation on Combined Dataset")
println("Device: ", CUDA.functional() ? "CUDA GPU ($(CUDA.device()))" : "CPU")
println("="^60)

# 1. Load the first 12,000 events from each dataset
println("Loading datasets (first 12,000 events each)...")
sm_path  = joinpath(@__DIR__, "wgamma_master_dataset.csv")
pos_path = joinpath(@__DIR__, "interference_positive.csv")
neg_path = joinpath(@__DIR__, "interference_negative.csv")

df_sm  = DataFrame(CSV.File(sm_path, limit=12000))
df_pos = DataFrame(CSV.File(pos_path, limit=12000))
df_neg = DataFrame(CSV.File(neg_path, limit=12000))

# Assign labels: 1 = SM, 2 = +ve, 3 = -ve
df_sm[!, :dataset]   .= "SM"
df_sm[!, :class_idx] .= 1

df_pos[!, :dataset]   .= "+ve"
df_pos[!, :class_idx] .= 2

df_neg[!, :dataset]   .= "-ve"
df_neg[!, :class_idx] .= 3

println("  SM events:       ", nrow(df_sm))
println("  +ve events:      ", nrow(df_pos))
println("  -ve events:      ", nrow(df_neg))

# Combine datasets
df = vcat(df_sm, df_pos, df_neg, cols=:union)
n_events = nrow(df)
println("Combined dataset size: $n_events events")

# 2. Input Features
feat_cols = [:lep_pt, :lep_eta, :lep_phi, :lep_charge, :lep_flavour,
             :pho_pt, :pho_eta, :pho_phi, :met, :met_phi]
println("Features (10): ", join(string.(feat_cols), ", "))

function get_feature_matrix(sub_df, cols)
    mat = Matrix{Float32}(undef, length(cols), nrow(sub_df))
    for (j, col) in enumerate(cols)
        mat[j, :] = Float32.(sub_df[!, col])
    end
    return mat
end

# 3. Stratified 5-Fold Assignment
Random.seed!(67)
folds = zeros(Int, n_events)
for c in 1:3
    idx = findall(df.class_idx .== c)
    folds[idx] = shuffle((0:length(idx)-1) .% 5 .+ 1)
end
df[!, :cv_fold] = folds

for k in 1:5
    n_k = sum(folds .== k)
    n_sm = sum((folds .== k) .& (df.class_idx .== 1))
    n_p  = sum((folds .== k) .& (df.class_idx .== 2))
    n_m  = sum((folds .== k) .& (df.class_idx .== 3))
    println("  Fold $k: $n_k events (SM=$n_sm, +ve=$n_p, -ve=$n_m)")
end

# 4. Storage for Out-Of-Fold Predictions
P_SM    = zeros(Float32, n_events)
P_plus  = zeros(Float32, n_events)
P_minus = zeros(Float32, n_events)
O_NN    = zeros(Float32, n_events)

models = Any[]
scalers = Tuple{Matrix{Float32}, Matrix{Float32}}[]

loss_fn(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)

const epochs = 120
const batch_size = 150
const lr = 2e-3

println("\nStarting 5-Fold Cross Validation ($epochs epochs/fold, lr=$lr, batchsize=$batch_size)...")
p = Progress(5 * epochs; desc="Training Folds: ")

for k in 1:5
    train_mask = folds .!= k
    test_mask  = folds .== k

    train_df = df[train_mask, :]
    test_df  = df[test_mask, :]

    X_train = get_feature_matrix(train_df, feat_cols)
    Y_train = Flux.onehotbatch(train_df.class_idx, 1:3)
    X_test  = get_feature_matrix(test_df, feat_cols)

    # Standardize features using training fold statistics only
    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2)
    σ[σ .== 0] .= 1.0f0

    X_train_norm = (X_train .- μ) ./ σ
    X_test_norm  = (X_test .- μ) ./ σ

    # Define model architecture (10 -> 64 -> 128 -> 64 -> 3)
    model = Chain(
        Dense(10 => 64, relu),
        Dense(64 => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => 3)
    ) |> dev

    opt = Flux.setup(Adam(lr), model)
    loader = Flux.DataLoader((X_train_norm, Y_train), batchsize=batch_size, shuffle=true) |> dev

    for ep in 1:epochs
        Flux.train!(loss_fn, model, loader, opt)
        next!(p)
    end

    # Out-of-fold inference
    logits = model(X_test_norm |> dev) |> cpu_dev
    probs  = softmax(logits, dims=1)

    P_SM[test_mask]    .= probs[1, :]
    P_plus[test_mask]  .= probs[2, :]
    P_minus[test_mask] .= probs[3, :]
    O_NN[test_mask]    .= probs[2, :] .- probs[3, :]

    push!(models, cpu_dev(deepcopy(model)))
    push!(scalers, (μ, σ))
end

println("\nTraining complete! Calculating diagnostics...")

# 5. ML Diagnostics
pred_classes = [argmax([P_SM[i], P_plus[i], P_minus[i]]) for i in 1:n_events]
overall_acc = mean(pred_classes .== df.class_idx) * 100.0
println(@sprintf("Overall 3-Class Accuracy: %.2f%%", overall_acc))

println("\nO_NN Statistics by Class:")
for (c, name) in [(1, "SM"), (2, "+ve"), (3, "-ve")]
    mask = df.class_idx .== c
    o_c = O_NN[mask]
    println(@sprintf("  %-5s: mean = %+.4f, std = %.4f, median = %+.4f, [min, max] = [%+.4f, %+.4f]",
                     name, mean(o_c), std(o_c), median(o_c), minimum(o_c), maximum(o_c)))
end

# Check probability sum
prob_sums = P_SM .+ P_plus .+ P_minus
max_dev = maximum(abs.(prob_sums .- 1.0f0))
println(@sprintf("Max deviation of (P_SM + P_+ + P_- - 1): %.2e", max_dev))

# 6. Save Predictions Dataset
df[!, :P_SM]    = P_SM
df[!, :P_plus]  = P_plus
df[!, :P_minus] = P_minus
df[!, :O_NN]    = O_NN

out_csv = joinpath(@__DIR__, "wgamma_combined_predictions.csv")
CSV.write(out_csv, df)
println("\nSaved out-of-fold predictions to: $out_csv")

# 7. Generate Figure: O_NN Distributions
bins_nn = range(-1.0, 1.0, length=41)

p_dist = stephist(O_NN[df.class_idx .== 1], bins=bins_nn, normalize=:pdf,
    label="SM", color=:green, lw=2,
    xlabel="O_NN = P_+ - P_-", ylabel="Probability Density",
    title="O_NN Distribution (5-Fold Cross-Validation, Combined Dataset)",
    legend=:topright)

stephist!(p_dist, O_NN[df.class_idx .== 2], bins=bins_nn, normalize=:pdf,
    label="+ve interference", color=:red, lw=2)

stephist!(p_dist, O_NN[df.class_idx .== 3], bins=bins_nn, normalize=:pdf,
    label="-ve interference", color=:blue, lw=2)

fig_path = joinpath(@__DIR__, "Figure_4_combined.png")
savefig(p_dist, fig_path)
println("Saved distribution plot to: $fig_path")

# 8. Figure 4: Differential cross section vs O_NN (Physical Observable dσ/dO_NN)
# Exactly matching run_wgamma.jl using physics weights w_SM and w_int
println("\nGenerating physics differential cross section plot (SM/20 vs c_ΦWB/Λ²)...")
df_full_master = DataFrame(CSV.File(sm_path))
n_master = nrow(df_full_master)
X_master = get_feature_matrix(df_full_master, feat_cols)

O_NN_full = zeros(Float32, n_master)
n_oof = min(12000, n_master)
O_NN_full[1:n_oof] .= O_NN[1:n_oof]

if n_master > n_oof
    rem_idx = (n_oof + 1):n_master
    X_rem = X_master[:, rem_idx]
    rem_preds = zeros(Float32, length(rem_idx))
    for (m, (μ_k, σ_k)) in zip(models, scalers)
        X_rem_norm = (X_rem .- μ_k) ./ σ_k
        probs_k = softmax(m(X_rem_norm), dims=1)
        rem_preds .+= (probs_k[2, :] .- probs_k[3, :])
    end
    O_NN_full[rem_idx] .= rem_preds ./ length(models)
end

scale_phys = 294.0 / sum(df_full_master.w_SM)
bins_phys = range(-1.0, 1.0, length=21)
bin_w_phys = step(bins_phys)
w_sm_phys = (df_full_master.w_SM ./ 20.0) .* (scale_phys / bin_w_phys)
w_int_phys = df_full_master.w_int .* (scale_phys / bin_w_phys)

p4 = stephist(O_NN_full, weights=weights(w_sm_phys), bins=bins_phys,
    label="SM/20 (c_ΦWB model)", color=:green, lw=1.5,
    xlabel="O_NN", ylabel="dσ/dO_NN [fb]",
    title="inclusive Wγ production", legend=:topright)
stephist!(p4, O_NN_full, weights=weights(w_int_phys), bins=bins_phys,
    label="c_ΦWB/Λ² = 1 TeV⁻²", color=:red, lw=1.5)

fig4_path = joinpath(@__DIR__, "Figure_4.png")
savefig(p4, fig4_path)
fig4_phys = joinpath(@__DIR__, "Figure_4_differential_xsec.png")
savefig(p4, fig4_phys)
println("Saved physics differential cross section plot to: $fig4_path (and $fig4_phys)")

println("="^60)
println("Done successfully!")
