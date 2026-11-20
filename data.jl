using CSV, DataFrames, Statistics, Printf

for (name, file) in [("SM ", "wgamma_master_dataset.csv"),
                     ("+ve", "interference_positive.csv"),
                     ("-ve", "interference_negative.csv")]
    d = DataFrame(CSV.File(file, limit=12000))
    println("=== $name (n=$(nrow(d))) ===")
    for col in [:pho_pt, :lep_pt, :met]
        v = Float64.(d[!, col])
        println(@sprintf("  %-7s: median=%6.1f, p90=%6.1f, max=%7.1f", col, median(v), quantile(v, 0.9), maximum(v)))
    end
    println("  lep_charge unique: ", sort(unique(d.lep_charge)), " | lep_flavour unique: ", sort(unique(d.lep_flavour)))
end