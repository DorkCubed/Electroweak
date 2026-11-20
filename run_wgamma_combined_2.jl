using Flux
using Plots, ProgressMeter
using CUDA
using CSV, DataFrames, Random, Statistics, StatsBase, Printf

const dev = CUDA.functional() ? Flux.gpu_device() : Flux.cpu_device()
const cpu_dev = Flux.cpu_device()

println("="^60)
println("Wγ 5-Fold Cross Validation on Combined Dataset (Tuned + CP-Augmented)")
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

# Diagnostic: check whether interference files carry varying event weights
for (name, sub) in [("SM (w_SM)", df_sm), ("+ve (w_int)", df_pos), ("-ve (w_int)", df_neg)]
    col = startswith(name, "SM") ? :w_SM : :w_int
    if hasproperty(sub, col)
        vals = abs.(Float64.(sub[!, col]))
        cv = mean(vals) > 0 ? std(vals) / mean(vals) : 0.0
        println(@sprintf("  Weight check %-12s: min=%.3e, median=%.3e, max=%.3e, std/mean=%.3f",
                         name, minimum(vals), median(vals), maximum(vals), cv))
    end
end

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

# 2. Input Features & Physics Preprocessing
feat_cols = [:lep_pt, :lep_eta, :lep_phi, :lep_charge, :lep_flavour,
             :pho_pt, :pho_eta, :pho_phi, :met, :met_phi]
println("Raw features (10): ", join(string.(feat_cols), ", "))

function get_feature_matrix(sub_df, cols)
    mat = Matrix{Float32}(undef, length(cols), nrow(sub_df))
    for (j, col) in enumerate(cols)
        mat[j, :] = Float32.(sub_df[!, col])
    end
    return mat
end

@inline wrap_pi(θ::Float32) = mod(θ + Float32(π), Float32(2π)) - Float32(π)

# Physical CP transformation applied BEFORE feature encoding and scaling
function cp_transform_raw(X::Matrix{Float32})
    Xcp = copy(X)
    Xcp[2, :]  .= .-X[2, :]                        # lep_eta -> -lep_eta
    Xcp[3, :]  .= wrap_pi.(X[3, :] .+ Float32(π))  # lep_phi -> wrap(lep_phi + π)
    Xcp[4, :]  .= .-X[4, :]                        # lep_charge -> -lep_charge
    Xcp[7, :]  .= .-X[7, :]                        # pho_eta -> -pho_eta
    Xcp[8, :]  .= wrap_pi.(X[8, :] .+ Float32(π))  # pho_phi -> wrap(pho_phi + π)
    Xcp[10, :] .= wrap_pi.(X[10, :] .+ Float32(π)) # met_phi -> wrap(met_phi + π)
    return Xcp
end

@inline cp_class(c::Int) = c == 1 ? 1 : (c == 2 ? 3 : 2)

# Smooth encoding of the 10 raw features into 13 inputs:
# - log1p on heavy-tailed positive momenta (lep_pt, pho_pt, met)
# - (sin, cos) on periodic azimuthal angles to remove the ±π discontinuity
function encode_features(X::Matrix{Float32})
    n = size(X, 2)
    out = Matrix{Float32}(undef, 13, n)
    out[1, :]  .= log1p.(max.(X[1, :], 0.0f0))
    out[2, :]  .= X[2, :]
    out[3, :]  .= sin.(X[3, :])
    out[4, :]  .= cos.(X[3, :])
    out[5, :]  .= X[4, :]
    out[6, :]  .= X[5, :]
    out[7, :]  .= log1p.(max.(X[6, :], 0.0f0))
    out[8, :]  .= X[7, :]
    out[9, :]  .= sin.(X[8, :])
    out[10, :] .= cos.(X[8, :])
    out[11, :] .= log1p.(max.(X[9, :], 0.0f0))
    out[12, :] .= sin.(X[10, :])
    out[13, :] .= cos.(X[10, :])
    return out
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
    n_k  = sum(folds .== k)
    n_sm = sum((folds .== k) .& (df.class_idx .== 1))
    n_p  = sum((folds .== k) .& (df.class_idx .== 2))
    n_m  = sum((folds .== k) .& (df.class_idx .== 3))
    println("  Fold $k: $n_k events (SM=$n_sm, +ve=$n_p, -ve=$n_m)")
end

# 4. Storage for Out-Of-Fold Predictions & Diagnostics
P_SM       = zeros(Float32, n_events)
P_plus     = zeros(Float32, n_events)
P_minus    = zeros(Float32, n_events)
O_NN       = zeros(Float32, n_events)
P_SM_cp    = zeros(Float32, n_events)
O_NN_cp    = zeros(Float32, n_events)

models  = Any[]
scalers = Tuple{Matrix{Float32}, Matrix{Float32}}[]
all_train_losses = Vector{Float32}[]
all_val_losses   = Vector{Float32}[]

loss_fn(m, x, y) = Flux.Losses.logitcrossentropy(m(x), y)

const max_epochs   = 200
const patience     = 18
const lr_patience  = 6
const batch_size   = 128
const lr_init      = 1.0f-3
const lr_min       = 1.0f-5
const weight_decay = 1.0f-4
const val_fraction = 0.10

println("\nStarting 5-Fold CV (max_epochs=$max_epochs, patience=$patience, lr=$lr_init, wd=$weight_decay, batch=$batch_size, no dropout)...")

for k in 1:5
    dev_mask  = folds .!= k
    test_mask = folds .== k

    dev_df  = df[dev_mask, :]
    test_df = df[test_mask, :]

    # Stratified 90/10 split of development events into sub-train and internal validation
    # Done BEFORE CP augmentation so an event and its CP copy never split across train/val
    n_dev = nrow(dev_df)
    is_val = falses(n_dev)
    for c in 1:3
        c_idx = shuffle(findall(dev_df.class_idx .== c))
        n_val_c = max(1, round(Int, val_fraction * length(c_idx)))
        is_val[c_idx[1:n_val_c]] .= true
    end

    sub_train_df = dev_df[.!is_val, :]
    val_df       = dev_df[is_val, :]

    # Raw physical features
    X_sub_raw  = get_feature_matrix(sub_train_df, feat_cols)
    y_sub_raw  = Int.(sub_train_df.class_idx)
    X_val_raw  = get_feature_matrix(val_df, feat_cols)
    y_val_raw  = Int.(val_df.class_idx)
    X_test_raw = get_feature_matrix(test_df, feat_cols)

    # CP-augment the sub-training set only
    X_sub_aug_raw = hcat(X_sub_raw, cp_transform_raw(X_sub_raw))
    y_sub_aug     = vcat(y_sub_raw, cp_class.(y_sub_raw))

    # Encode features (log1p on pT/MET, sin/cos on φ)
    X_train_enc   = encode_features(X_sub_aug_raw)
    X_val_enc     = encode_features(X_val_raw)
    X_test_enc    = encode_features(X_test_raw)
    X_test_cp_enc = encode_features(cp_transform_raw(X_test_raw))

    # Standardize using sub-training fold statistics only
    μ = mean(X_train_enc, dims=2)
    σ = std(X_train_enc, dims=2)
    σ[σ .== 0] .= 1.0f0

    X_train_norm   = (X_train_enc .- μ) ./ σ
    X_val_norm     = (X_val_enc .- μ) ./ σ
    X_test_norm    = (X_test_enc .- μ) ./ σ
    X_test_cp_norm = (X_test_cp_enc .- μ) ./ σ

    Y_train = Flux.onehotbatch(y_sub_aug, 1:3)
    Y_val   = Flux.onehotbatch(y_val_raw, 1:3)

    # Smooth MLP without Dropout (mild AdamW weight decay + early stopping prevents saturation
    # without blurring the SM vs. interference decision boundary)
    model = Chain(
        Dense(13 => 128, swish),
        Dense(128 => 128, swish),
        Dense(128 => 64, swish),
        Dense(64 => 3)
    ) |> dev

    current_lr = lr_init
    opt = Flux.setup(AdamW(current_lr, (0.9f0, 0.999f0), weight_decay), model)
    loader = Flux.DataLoader((X_train_norm, Y_train), batchsize=batch_size, shuffle=true) |> dev

    X_val_dev = X_val_norm |> dev
    Y_val_dev = Y_val |> dev

    best_val_loss = Inf32
    best_epoch    = 0
    best_model    = deepcopy(cpu_dev(model))
    wait_count    = 0
    lr_wait       = 0

    fold_train_losses = Float32[]
    fold_val_losses   = Float32[]

    p = Progress(max_epochs; desc="Fold $k/5: ")
    for ep in 1:max_epochs
        Flux.train!(loss_fn, model, loader, opt)

        tr_loss = 0.0f0
        n_tr    = 0
        for (xb, yb) in loader
            bs = size(xb, 2)
            tr_loss += loss_fn(model, xb, yb) * bs
            n_tr += bs
        end
        tr_loss /= n_tr
        vl_loss = loss_fn(model, X_val_dev, Y_val_dev)

        push!(fold_train_losses, Float32(tr_loss))
        push!(fold_val_losses, Float32(vl_loss))

        if vl_loss < best_val_loss - 1.0f-5
            best_val_loss = Float32(vl_loss)
            best_epoch    = ep
            best_model    = deepcopy(cpu_dev(model))
            wait_count    = 0
            lr_wait       = 0
        else
            wait_count += 1
            lr_wait    += 1
            if lr_wait >= lr_patience && current_lr > lr_min
                current_lr = max(current_lr * 0.5f0, lr_min)
                Flux.adjust!(opt, current_lr)
                lr_wait = 0
            end
        end

        next!(p; showvalues=[(:train_loss, @sprintf("%.4f", tr_loss)),
                             (:val_loss,   @sprintf("%.4f", vl_loss)),
                             (:best_val,   @sprintf("%.4f (ep %d)", best_val_loss, best_epoch)),
                             (:lr,         @sprintf("%.1e", current_lr))])

        if wait_count >= patience
            finish!(p)
            break
        end
    end

    println(@sprintf("  Fold %d stopped: best val_loss = %.4f at epoch %d", k, best_val_loss, best_epoch))
    push!(all_train_losses, fold_train_losses)
    push!(all_val_losses, fold_val_losses)

    # Out-of-fold inference using best restored model
    best_dev = best_model |> dev

    logits = best_dev(X_test_norm |> dev) |> cpu_dev
    probs  = softmax(logits, dims=1)

    P_SM[test_mask]    .= probs[1, :]
    P_plus[test_mask]  .= probs[2, :]
    P_minus[test_mask] .= probs[3, :]
    O_NN[test_mask]    .= probs[2, :] .- probs[3, :]

    # Held-out CP-consistency diagnostic (not added to final dataset)
    logits_cp = best_dev(X_test_cp_norm |> dev) |> cpu_dev
    probs_cp  = softmax(logits_cp, dims=1)
    P_SM_cp[test_mask] .= probs_cp[1, :]
    O_NN_cp[test_mask] .= probs_cp[2, :] .- probs_cp[3, :]

    push!(models, best_model)
    push!(scalers, (μ, σ))
end

println("\nTraining complete! Calculating diagnostics...")

# 5. ML Diagnostics
pred_classes = [argmax([P_SM[i], P_plus[i], P_minus[i]]) for i in 1:n_events]
overall_acc = mean(pred_classes .== df.class_idx) * 100.0
println(@sprintf("Overall 3-Class Accuracy: %.2f%%", overall_acc))

println("\nO_NN & P_SM Statistics by Class:")
for (c, name) in [(1, "SM"), (2, "+ve"), (3, "-ve")]
    mask = df.class_idx .== c
    o_c = O_NN[mask]
    p_sm_c = P_SM[mask]
    edge_frac = mean(abs.(o_c) .> 0.9f0) * 100.0
    println(@sprintf("  %-5s: mean(O_NN)=%+.4f, std=%.4f, mean(P_SM)=%.4f, |O_NN|>0.9=%.2f%%",
                     name, mean(o_c), std(o_c), mean(p_sm_c), edge_frac))
end

# CP consistency diagnostic on held-out events
cp_o_err  = mean(abs.(O_NN .+ O_NN_cp))
cp_sm_err = mean(abs.(P_SM .- P_SM_cp))
println(@sprintf("\nCP Consistency (held-out): mean |O_NN(x) + O_NN(x^CP)| = %.4f, mean |P_SM(x) - P_SM(x^CP)| = %.4f",
                 cp_o_err, cp_sm_err))

# Check probability sum
prob_sums = P_SM .+ P_plus .+ P_minus
max_dev = maximum(abs.(prob_sums .- 1.0f0))
println(@sprintf("Max deviation of (P_SM + P_+ + P_- - 1): %.2e", max_dev))

# 6. Save Loss Curves Diagnostic Plot
p_loss = plot(xlabel="Epoch", ylabel="Cross-Entropy Loss",
              title="5-Fold Training & Validation Loss", legend=:topright)
for k in 1:5
    plot!(p_loss, all_train_losses[k], lw=1.2, ls=:dash, alpha=0.6, label=(k == 1 ? "Train (folds 1-5)" : ""))
    plot!(p_loss, all_val_losses[k],   lw=1.8, alpha=0.9, label=(k == 1 ? "Val (folds 1-5)" : ""))
end
loss_fig_path = joinpath(@__DIR__, "Figure_4_loss_curves.png")
savefig(p_loss, loss_fig_path)
println("Saved loss curves to: $loss_fig_path")

# 7. Save Predictions Dataset
df[!, :P_SM]    = P_SM
df[!, :P_plus]  = P_plus
df[!, :P_minus] = P_minus
df[!, :O_NN]    = O_NN

out_csv = joinpath(@__DIR__, "wgamma_combined_predictions.csv")
CSV.write(out_csv, df)
println("\nSaved out-of-fold predictions to: $out_csv")

# 8. Generate Figure: O_NN Distributions
bins_nn = range(-1.0, 1.0, length=41)

p_dist = stephist(O_NN[df.class_idx .== 1], bins=bins_nn, normalize=:pdf,
    label="SM", color=:green, lw=2,
    xlabel="O_NN = P_+ - P_-", ylabel="Probability Density",
    title="O_NN Distribution (5-Fold CV, Tuned + CP-Augmented)",
    legend=:topright)

stephist!(p_dist, O_NN[df.class_idx .== 2], bins=bins_nn, normalize=:pdf,
    label="+ve interference", color=:red, lw=2)

stephist!(p_dist, O_NN[df.class_idx .== 3], bins=bins_nn, normalize=:pdf,
    label="-ve interference", color=:blue, lw=2)

fig_path = joinpath(@__DIR__, "Figure_4_combined.png")
savefig(p_dist, fig_path)
println("Saved distribution plot to: $fig_path")

# 9. Figure 4: Differential cross section vs O_NN (Physical Observable dσ/dO_NN)
println("\nGenerating physics differential cross section plot (SM/20 vs c_ΦWB/Λ²)...")
df_full_master = DataFrame(CSV.File(sm_path))
n_master = nrow(df_full_master)

O_NN_full = zeros(Float32, n_master)
n_oof = min(nrow(df_sm), n_master)
O_NN_full[1:n_oof] .= O_NN[1:n_oof]

if n_master > n_oof
    rem_idx = (n_oof + 1):n_master
    X_rem_enc = encode_features(get_feature_matrix(df_full_master[rem_idx, :], feat_cols))
    rem_preds = zeros(Float32, length(rem_idx))
    for (m, (μ_k, σ_k)) in zip(models, scalers)
        X_rem_norm = (X_rem_enc .- μ_k) ./ σ_k
        probs_k = softmax(m(X_rem_norm), dims=1)
        rem_preds .+= (probs_k[2, :] .- probs_k[3, :])
    end
    O_NN_full[rem_idx] .= rem_preds ./ length(models)
end

scale_phys = 294.0 / sum(df_full_master.w_SM)
bins_phys  = range(-1.0, 1.0, length=21)
bin_w_phys = step(bins_phys)
w_sm_phys  = (df_full_master.w_SM ./ 20.0) .* (scale_phys / bin_w_phys)
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
