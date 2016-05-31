using ProgressMeter

"""
    dimension_test(numSamples,numFeatures,blockSize)

    # Description
    Throws an error if the dimensions of the particular problem instance do not
    make sense.
"""
function dimension_test(numSamples::Integer, numFeatures::Integer, blockSize::Integer)
    # PCA of a single vector makes no sense...
    if numSamples == 1
        error("StreamingPCA.jl: <<Dataset consists of a single sample.>>")
    end

    # PCA of samples with a single feature...
    if numFeatures == 1
        error("StreamingPCA.jl: <<Dataset consists of samples with a single feature.>>")
    end

    # Require that the size of the dataset be divisible by the block size
    if (numSamples % blockSize) != 0 
        error("StreamingPCA.jl:  <<Dataset size ($(numSamples)) must be divisible by block size ($(blockSize)).>>")
    end
end

"""
    block_ranges(numSamples::Integer,blockSize::Integer) 

    # Description
    Return a UnitRange{Int64} type for the dataset indicies of the
    specified batch.
"""
function block_range(numSamples::Integer,blockSize::Integer, batchNumber::Integer)
    loIdx = blockSize*(batchNumber-1) + 1
    hiIdx = loIdx + blockSize - 1

    return loIdx:hiIdx
end


"""
    Stream over samples, update the Q/R as needed if we have enough data to
    warrant a full batch
"""
function streaming_pca_process_sample!(x::Array{Float64}, 
                                       Q::Array{Float64}, R::Array{Float64},
                                       S::Array{Float64}, sBQ::Array{Float64},
                                       batchSize::Integer,sampleNumber::Integer)

    gemm!('T','N',1/batchSize,x,sBQ[sample,m:],1.0,S)
    
end

function streaming_pca_process_batch!(batch::Array{Float64},Q::Array{Float64},R::Array{Float64})
    numFeatures,batchSize = size(batch)
    rank = size(Q,2)
    S = zeros(numFeatures,rank)
    scale = 1/batchSize

    sBQ = scale.*(batch*Q)

    for sample in 1:batchSize
        gemm!('T','N',1.0,batch[sample,:],sBQ[sample,m:],1.0,S)
    end

    Qloc,Rloc = qr(S)
end


"""
   streaming_pca(X::Array{Float64}, blockSize::Integer)

   # Description
   Given a dataset X, where samples are arranged as rows, and a valid block 
   size, return the principal components of the dataset as estimated by the 
   'Block-Stochastic Orthogonal Iteration' given in Alg. 1 of
   (Mitliagkas, Caramanis & Jain, 2013).
"""
function streaming_pca(X::Array{Float64}, blockSize::Integer; rank::Integer = 1)
    numSamples, numFeatures = size(X)
    dimension_test(numSamples,numFeatures,blockSize)

    numBlocks = convert(Integer,numSamples / blockSize)

    # Initial Parameters
    Q, R = qr(randn(numFeatures,rank))
    scale = 1 / blockSize    

    S = []  # Declaring for access outside block    
    @showprogress 0.001 "Streaming batches..." 20 for block in 1:numBlocks        
        S = zeros(numFeatures,rank)
        # Get indicies for this block
        blockRange = block_range(numSamples,blockSize,block)
        sXQ = scale.*(X[blockRange,:]*Q)

        for sample in blockRange
            # Over samples, the core procedure is (assuming row
            # vector samples)...
            #     S = (a'aQ)/s + (b'bQ)/s + (c'cQ)/s + (d'dQ)/s + ...
            # So, we should be able to put together this operation with
            # just gemm, now that we have precomputed all (aQ)/s, (bQ)/s, ...
            gemm!('T','N',1.0,X[sample,:],sXQ[sample-blockRange[1]+1,:],1.0,S)
        end

        Q, R = qr(S)
    end

    # Now get the data projection
    XQ = (Q\(X'))'
    # What is the order of the components (via norm)?
    sq_pc_energy = vec(sum(XQ.^2,1));
    # Sort components by energy
    Q = Q[:,sortperm(sq_pc_energy;rev=true)]
    sort!(sq_pc_energy;rev=true)

    return Q,XQ,sq_pc_energy
end

