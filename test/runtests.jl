using StreamingPCA
using Base.Test

# Invalid Problem Dimensions
caught = false
try
    @test streaming_pca(randn(100,5),17)
catch
    caught = true
    println("[dim.1] Caught dimension error.")
end
if !caught
    error("Did not catch dimension error.")
end

caught = false
try
    @test streaming_pca(randn(1,100),17)
catch
    caught = true
    println("[dim.2] Caught dimension error.")
end
if !caught
    error("Did not catch dimension error.")
end
