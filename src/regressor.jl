type Regressor
    n_samples::Int
    n_features::Int
    n_max_features::Int
    improvements::Vector{Float64}
    oob_error::Float64
    trees::Vector{Tree}

    function Regressor(rf, x, y)
        n_samples, n_features = size(x)

        if n_samples != length(y)
            throw(DimensionMismatch(""))
        end

        n_max_features = resolve_max_features(rf.max_features, n_features)
        @assert 0 < n_max_features <= n_features

        improvements = zeros(Float64, n_features)
        trees = Array(Tree, rf.n_estimators)
        new(n_samples, n_features, n_max_features, improvements, NaN, trees)
    end
end

typealias RandomForestRegressor RandomForest{Regressor}

function RandomForestRegressor(;n_estimators::Int=10, max_features::Union{Integer,AbstractFloat,Symbol}=:third, max_depth=nothing, min_samples_split::Int=2)
    RandomForest{Regressor}(n_estimators, max_features, max_depth, min_samples_split, :mse)
end

tuple_add(x::Tuple, y::Tuple) = x[1] + y[1], x[2] + y[2]

function fit!{T<:TabularData}(rf::RandomForestRegressor, x::T, y::AbstractVector; do_oob=false)
    learner = Regressor(rf, x, y)
    n_samples = learner.n_samples

    # pre-allocation
    bootstrap = Array(Int, n_samples)
    sample_weight = Array(Float64, n_samples)
    oob_predict = zeros(n_samples)
    oob_count = zeros(Int, n_samples)

    learner.trees = @parallel (vcat) for b in 1:rf.n_estimators
        bootstrap = rand(1:n_samples, n_samples)
        sample_weight = sample_weights(bootstrap)
        example = Trees.Example{T}(x, y, sample_weight)
        tree = Trees.Tree()
        Trees.fit!(tree, example, rf.criterion, learner.n_max_features, rf.max_depth, rf.min_samples_split)
        if do_oob
            tree.sample_weights = sample_weight
        end
        tree
    end
    if do_oob
        oob = @parallel (tuple_add) for s in 1:n_samples
            oob_predict = 0.
            oob_count = 0
            for tree in learner.trees
                if tree.sample_weights[s] == 0.0
                    oob_predict += Trees.predict(tree, vec(x[s, :]))
                    oob_count += 1
                end
            end
            if oob_count == 0
                return 0, 0
            end
            (y[s] - oob_predict / oob_count)^2, (oob_count > 0 ? 1. : 0.)
        end
        learner.oob_error = sqrt(oob[1] / oob[2])
    end

    set_improvements!(learner)
    rf.learner = learner
    return
end

function predict{T<:TabularData}(rf::RandomForestRegressor, x::T)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    n_samples = size(x, 1)
    output = Array(Float64, n_samples)
    vs = Array(Float64, rf.n_estimators)

    for i in 1:n_samples
        for b in 1:rf.n_estimators
            tree = rf.learner.trees[b]
            vs[b] = Trees.predict(tree, vec(x[i, :]))
        end
        output[i] = mean(vs)
    end

    output
end

function oob_error(rf::RandomForestRegressor)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    rf.learner.oob_error
end
