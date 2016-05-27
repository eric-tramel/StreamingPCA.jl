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
    hiIdx = loIdx + blockSize

    return loIdx:hiIdx
end


"""
   streaming_pca(X::Array{Float64}, blockSize::Integer)

   # Description
   Given a dataset X, where samples are arranged as rows, and a valid block 
   size, return the principal components of the dataset as estimated by the 
   'Block-Stochastic Orthogonal Iteration' given in Alg. 1 of
   (Mitliagkas, Caramanis & Jain, 2013).
"""
function streaming_pca(X::Array{Float64}, blockSize::Integer)
    numSamples, numFeatures = size(X)
    dimension_test(numSamples,numFeatures,blockSize)

    numBlocks = convert(Integer,numSamples / blockSize)

end

