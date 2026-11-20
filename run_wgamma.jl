using Flux
using Plots, ProgressMeter
using CUDA
using CSV, DataFrames, Random, Statistics, StatsBase

const dev = CUDA.functional() ? Flux.gpu_device() : Flux.cpu_device()
const cpu_dev = Flux.cpu_device()

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

const epochs = 100
p = Progress(5 * epochs)

for k in 1:5
    model = Chain(
        Dense(10 => 64, relu),
        Dense(64 => 128, relu),
        Dense(128 => 64, relu),
        Dense(64 => 3)
    ) |> dev

    opt = Flux.setup(Adam(8e-3), model)

    train_mask = folds .!= k
    test_mask = folds .== k

    train_df = df[train_mask, :]
    test_df = df[test_mask, :]

    W_sm = sum(train_df.w_SM)
    W_pos = sum(abs, train_df.w_int[train_df.w_int.>0])
    W_neg = sum(abs, train_df.w_int[train_df.w_int.<0])

    train_aug = DataFrame(x=Vector{Float32}[], y=Int[], w=Float32[])
    for row in eachrow(train_df)
        x = Float32[row[col] for col in feat_cols]
        xcp = cp_transform(x)
        a = Float32(row.w_SM / W_sm)
        wint = Float32(row.w_int)
        b = abs(wint) / Float32(wint > 0 ? W_pos : W_neg)
        t_orig = wint > 0 ? 2 : 3
        t_cp = wint > 0 ? 3 : 2

        push!(train_aug, [x, 1, a])
        push!(train_aug, [x, t_orig, b])
        push!(train_aug, [xcp, 1, a])
        push!(train_aug, [xcp, t_cp, b])
    end

    X_train = reduce(hcat, train_aug.x)
    Y_train = Flux.onehotbatch(train_aug.y, 1:3)
    W_train = train_aug.w

    μ = mean(X_train, dims=2)
    σ = std(X_train, dims=2)
    σ[σ.==0] .= 1.0

    X_train_norm = (X_train .- μ) ./ σ

    X_test = reduce(hcat, [Float32[row[col] for col in feat_cols] for row in eachrow(test_df)])
    X_test_norm = (X_test .- μ) ./ σ

    n_aug = size(X_train_norm, 2)
    sub_n = div(n_aug, 2)
    for _ in 1:epochs
        idx = randperm(n_aug)[1:sub_n]
        loader = Flux.DataLoader((X_train_norm[:, idx], Y_train[:, idx], W_train[idx]), batchsize=64, shuffle=true) |> dev
        Flux.train!(loss_fn, model, loader, opt)

        next!(p)
    end

    probs = softmax(model(X_test_norm |> dev)) |> cpu_dev
    view(P_SM, test_mask) .= probs[1, :]
    view(P_plus, test_mask) .= probs[2, :]
    view(P_minus, test_mask) .= probs[3, :]
    view(O_NN, test_mask) .= probs[2, :] .- probs[3, :]
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
