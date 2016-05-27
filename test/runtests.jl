using StreamingPCA
using Base.Test

# Invalid Problem Dimensions
caught = false
try
    streaming_pca(randn(100,5),17)
catch
    caught = true
    println("[dim.1] Caught dimension error. [OK]")
end
if !caught
    error("[dim.1] Did not catch dimension error. [BAD]")
end

caught = false
try
    streaming_pca(randn(1,100),17)
catch
    caught = true
    println("[dim.2] Caught dimension error. [OK]")
end
if !caught
    error("[dim.1] Did not catch dimension error. [BAD]")
end

caught = false
try
    streaming_pca(randn(100,1),17)
catch
    caught = true
    println("[dim.3] Caught dimension error. [OK]")
end
if !caught
    error("[dim.1] Did not catch dimension error. [BAD]")
end

# Block ranges
r = StreamingPCA.block_range(100,10,1)
if r != 1:10
    println("[ran.1] Invalid block range. [OK]")
end


# Valid problem
try 
    streaming_pca(randn(100,5),10)
catch
    error("[gen.1] Valid call threw error. [BAD]")
end
println("[gen.1] Valid call is valid. [OK]")