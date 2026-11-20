using Flux
using Plots, ProgressMeter
using CSV, DataFrames, Random, Statistics, StatsBase

# 1. Load data and setup bins
df = DataFrame(CSV.File(joinpath(@__DIR__, "wgamma_master_dataset.csv")))
scale = 294.0 / sum(df.w_SM)

# Figure 2: Differential cross section vs delta_phi(l, gamma)
bins_phi = range(-Float64(π), Float64(π), length=21)
bin_w_phi = step(bins_phi)
w_sm_phi = (df.w_SM ./ 20.0) .* (scale / bin_w_phi)
w_int_phi = df.w_int .* (scale / bin_w_phi)

#=
p2 = stephist(df.signed_dphi_lg, weights=weights(w_sm_phi), bins=bins_phi,
    label="SM/20", color=:blue, lw=1.5,
    xlabel="Δϕ_lγ [rad]", ylabel="dσ/dΔϕ_lγ [fb/rad]",
    title="inclusive Wγ production", legend=:bottomleft)
stephist!(p2, df.signed_dphi_lg, weights=weights(w_int_phi), bins=bins_phi,
    label="c_ΦWB/Λ² = 1 TeV⁻²", color=:red, lw=1.5)
savefig(p2, joinpath(@__DIR__, "Figure_2.png"))
=#

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

# 3. 5-Fold Cross Validation
Random.seed!(67)
n_events = nrow(df)
folds = shuffle(1:n_events) .% 5 .+ 1

P_SM = zeros(Float32, n_events)
P_plus = zeros(Float32, n_events)
P_minus = zeros(Float32, n_events)
O_NN = zeros(Float32, n_events)

loss_fn(m, x, y, w) = sum(w .* vec(Flux.Losses.logitcrossentropy(m(x), y; agg=identity))) / sum(w)

const max_epochs = 200
const patience = 15

function build_aug(sub_df, feat_cols, W_sm, W_pos, W_neg)
    xs, ys, ws = Vector{Float32}[], Int[], Float32[]
    for row in eachrow(sub_df)
        x = Float32[row[col] for col in feat_cols]
        xcp = cp_transform(x)
        a = Float32(row.w_SM / W_sm)
        wint = Float32(row.w_int)
        b = abs(wint) / Float32(wint > 0 ? W_pos : W_neg)
        t_orig = wint > 0 ? 2 : 3
        t_cp = wint > 0 ? 3 : 2
        push!(xs, x)
        push!(ys, 1)
        push!(ws, a)
        push!(xs, x)
        push!(ys, t_orig)
        push!(ws, b)
        push!(xs, xcp)
        push!(ys, 1)
        push!(ws, a)
        push!(xs, xcp)
        push!(ys, t_cp)
        push!(ws, b)
    end
    return reduce(hcat, xs), Flux.onehotbatch(ys, 1:3), Float32.(ws)
end

p = Progress(5; desc="Folds")

for k in 1:5
    model = Chain(
        Dense(10 => 128, relu),
        Dense(128 => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => 3)
    )
    opt = Flux.setup(Adam(1e-2), model)

    dev_mask = folds .!= k
    test_mask = folds .== k
    dev_idx = findall(dev_mask)

    # sub-training / validation split (80/20) on original physical events
    n_dev = length(dev_idx)
    perm = shuffle(1:n_dev)
    n_sub = round(Int, 0.8 * n_dev)
    sub_idx = dev_idx[perm[1:n_sub]]
    val_idx = dev_idx[perm[n_sub+1:end]]

    sub_df = df[sub_idx, :]
    val_df = df[val_idx, :]
    test_df = df[test_mask, :]

    # normalisations from sub-training physical events only
    W_sm = sum(sub_df.w_SM)
    W_pos = sum(abs, sub_df.w_int[sub_df.w_int.>0])
    W_neg = sum(abs, sub_df.w_int[sub_df.w_int.<0])

    X_tr, Y_tr, W_tr = build_aug(sub_df, feat_cols, W_sm, W_pos, W_neg)

    # validation normalisations
    W_sm_v = sum(val_df.w_SM)
    W_pos_v = sum(abs, val_df.w_int[val_df.w_int.>0])
    W_neg_v = sum(abs, val_df.w_int[val_df.w_int.<0])
    X_val, Y_val, W_val = build_aug(val_df, feat_cols, W_sm_v, W_pos_v, W_neg_v)

    # standardise using sub-training stats
    μ = mean(X_tr, dims=2)
    σ = std(X_tr, dims=2)
    σ[σ.==0] .= 1.0f0
    X_tr .= (X_tr .- μ) ./ σ
    X_val_n = (X_val .- μ) ./ σ

    X_test = reduce(hcat, [Float32[row[col] for col in feat_cols] for row in eachrow(test_df)])
    X_test_n = (X_test .- μ) ./ σ

    # training with early stopping
    best_loss = Inf
    wait = 0
    best_state = Flux.state(model)
    for ep in 1:max_epochs
        loader = Flux.DataLoader((X_tr, Y_tr, W_tr), batchsize=128, shuffle=true)
        Flux.train!(loss_fn, model, loader, opt)

        vl = loss_fn(model, X_val_n, Y_val, W_val)
        if vl < best_loss
            best_loss = vl
            wait = 0
            best_state = Flux.state(model)
        else
            wait += 1
            wait >= patience && break
        end
    end
    Flux.loadmodel!(model, best_state)

    probs = softmax(model(X_test_n))
    view(P_SM, test_mask) .= probs[1, :]
    view(P_plus, test_mask) .= probs[2, :]
    view(P_minus, test_mask) .= probs[3, :]
    view(O_NN, test_mask) .= probs[2, :] .- probs[3, :]
    next!(p)
end

# 4. Figure 4: Differential cross section vs O_NN
bins_nn = range(-1.0, 1.0, length=21)
bin_w_nn = step(bins_nn)
w_sm_nn = (df.w_SM ./ 20.0) .* (scale / bin_w_nn)
w_int_nn = df.w_int .* (scale / bin_w_nn)

p4 = stephist(O_NN, weights=weights(w_sm_nn), bins=bins_nn,
    label="SM/20 (c_ΦWB model)", color=:green, lw=1.5,
    xlabel="O_NN", ylabel="dσ/dO_NN [fb]",
    title="inclusive Wγ production", legend=:topright)
stephist!(p4, O_NN, weights=weights(w_int_nn), bins=bins_nn,
    label="c_ΦWB/Λ² = 1 TeV⁻²", color=:red, lw=1.5)
savefig(p4, joinpath(@__DIR__, "Figure_4.png"))

# 5. Save output dataset
df.cv_fold = folds
df.P_SM = P_SM
df.P_plus = P_plus
df.P_minus = P_minus
df.O_NN = O_NN
CSV.write(joinpath(@__DIR__, "wgamma_predictions.csv"), df)
