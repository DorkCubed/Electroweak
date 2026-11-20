using Pkg
if !haskey(Pkg.project().dependencies, "Plots")
    Pkg.activate(joinpath(DEPOT_PATH[1], "environments", "v1.12"))
end

using CSV, DataFrames, Plots, StatsBase

println("Loading predictions from wgamma_combined_predictions.csv...")
pred_path = joinpath(@__DIR__, "wgamma_combined_predictions.csv")
df = DataFrame(CSV.File(pred_path))

# Filter for the SM dataset events (first 12,000 events from wgamma_master_dataset.csv)
df_sm = df[df.dataset .== "SM", :]
println("Found $(nrow(df_sm)) SM events with physics weights.")

# Cross-section scaling exactly like run_wgamma.jl
# Total SM cross-section is normalized to 294.0 fb
scale = 294.0 / sum(df_sm.w_SM)

bins_nn = range(-1.0, 1.0, length=21)
bin_w_nn = step(bins_nn)
w_sm_nn  = (df_sm.w_SM ./ 20.0) .* (scale / bin_w_nn)
w_int_nn = df_sm.w_int .* (scale / bin_w_nn)

# Figure 4: Differential cross section vs O_NN
p4 = stephist(df_sm.O_NN, weights=weights(w_sm_nn), bins=bins_nn,
    label="SM/20 (c_ΦWB model)", color=:green, lw=1.5,
    xlabel="O_NN", ylabel="dσ/dO_NN [fb]",
    title="inclusive Wγ production", legend=:topright)

stephist!(p4, df_sm.O_NN, weights=weights(w_int_nn), bins=bins_nn,
    label="c_ΦWB/Λ² = 1 TeV⁻²", color=:red, lw=1.5)

out_fig = joinpath(@__DIR__, "Figure_4_differential_xsec.png")
savefig(p4, out_fig)
println("Saved differential cross section plot to: $out_fig")

out_fig_std = joinpath(@__DIR__, "Figure_4.png")
savefig(p4, out_fig_std)
println("Also updated: $out_fig_std")

