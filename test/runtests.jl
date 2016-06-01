using StreamingPCA
using Base.Test
using MNIST
using MultivariateStats

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

# Make a test on MNIST
println("[mnist.1] Loading 60,000 data samples.")
X,lab = traindata()
X /= 255
rank = 10
batchSize = 100
println("[mnist.1] Running offline PCA.")
M = fit(PCA,X;maxoutdim=rank)
XrecOff = reconstruct(M,transform(M,X))
println("[mnist.1] Running `streaming_pca` for batch size $(batchSize).")
Q,XQ,pc_score = streaming_pca(X',batchSize;rank=rank)
println("[mnist.1] Principal Component size: $(size(Q))")
println("[mnist.1] Projected dataset size: $(size(XQ))")
println("[mnist.1] Prinicipal component scores: ")
println(pc_score'/maximum(pc_score))
Xrec = XQ*Q';
println("[mnist.1] Reconstructed MNIST dimensions : $(size(Xrec))")
fullmse = mean((X'-Xrec).^2)
fullmseOff  = mean((X-XrecOff).^2)
@printf("[mnist.1] Reconstructed MNIST MSE (offline): %e\n",fullmseOff)
@printf("[mnist.1] Reconstructed MNIST MSE (streaming): %e\n",fullmse)




