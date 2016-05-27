
"""
   streaming_pca(X::Array{Float64},blockSize::Integer)

   # Description
   Given a dataset X, where samples are arranged as rows, and a valid block 
   size, return the principal components of the dataset as estimated by the 
   'Block-Stochastic Orthogonal Iteration' given in Alg. 1 of
   (Mitliagkas, Caramanis & Jain, 2013).
"""
function streaming_pca(X::Array{Float64},blockSize::Integer)
    numSamples, numFeatures = size(X)
end