using MLBase

# internal data structure for a random forest classifier
type Classifier
    n_samples::Int
    n_features::Int
    n_max_features::Int
    label_mapping::LabelMap
    outtype::DataType
    improvements::Vector{Float64}
    oob_error::Float64
    trees::Vector{Tree}

    function Classifier(rf, x, y)
        n_samples, n_features = size(x)

        if n_samples != length(y)
            throw(DimensionMismatch(""))
        end

        n_max_features = resolve_max_features(rf.max_features, n_features)
        @assert 0 < n_max_features <= n_features

        label_mapping = labelmap(y)
        outtype = eltype(y)
        improvements = zeros(Float64, n_features)
        trees = Array(Tree, rf.n_estimators)
        new(n_samples, n_features, n_max_features, label_mapping, outtype, improvements, NaN, trees)
    end
end

typealias RandomForestClassifier RandomForest{Classifier}

function RandomForestClassifier(;n_estimators::Int=10, max_features::Union{Integer,AbstractFloat,Symbol}=:sqrt, max_depth=nothing, min_samples_split::Int=2, criterion::Symbol=:gini)
    if !(is(criterion, :gini) || is(criterion, :entropy))
        error("criterion is invalid (got: $criterion)")
    end
    RandomForest{Classifier}(n_estimators, max_features, max_depth, min_samples_split, criterion)
end


function fit!{T<:TabularData}(rf::RandomForestClassifier, x::T, y::AbstractVector; do_oob::Bool = false)
    learner = Classifier(rf, x, y)
    y_encoded = labelencode(learner.label_mapping, y)
    n_samples = learner.n_samples

    learner.trees = @parallel (vcat) for b in 1:rf.n_estimators
        tree = Trees.Tree()
        sample_weight = sampleweights(tree, n_samples)
        example = Trees.Example{T}(x, y_encoded, sample_weight)
        Trees.fit!(tree, example, rf.criterion, learner.n_max_features, rf.max_depth, rf.min_samples_split)
        tree
    end
    oob_error = nothing
    if do_oob
        oob_error = 0
        oob_error = @parallel (+) for tree in learner.trees
            sample_weight = sampleweights(tree, n_samples)
            hit = 0
            miss = 0
            for s in 1:n_samples
                if sample_weight[s] != 0.0
                    continue
                end

                # out-of-bag sample
                if Trees.predict(tree, vec(x[s, :])) == y_encoded[s]
                    hit += 1
                else
                    miss += 1
                end
            end
            miss / (hit + miss)
        end
        learner.oob_error = oob_error / rf.n_estimators
    end

    set_improvements!(learner)
    rf.learner = learner
    return
end

function predict{T<:TabularData}(rf::RandomForestClassifier, x::T)
    if is(rf.learner, nothing)
        error("not yet trained")
    end

    n_samples = size(x, 1)
    n_labels = length(rf.learner.label_mapping)

    output = @parallel (vcat) for i in 1:n_samples
        counts = zeros(Int, n_labels)
        for b in 1:rf.n_estimators
            tree = rf.learner.trees[b]
            vote = Trees.predict(tree, vec(x[i, :]))
            counts[vote] += 1
        end
        indmax(counts)
    end

    # TODO: use labeldecode method when MLBase.jl is released
    # labeldecode(rf.learner.label_mapping, output)
    predicted = Array(rf.learner.outtype, n_samples)
    for (i, o) in enumerate(output)
        predicted[i] = rf.learner.label_mapping.vs[o]
    end
    predicted
end

function normalize!(v)
    s = sum(v)
    for i in 1:endof(v)
        v[i] = v[i] / s
    end
    return
end

function oob_error(rf::RandomForestClassifier)
    if is(rf.learner, nothing)
        error("classifier not yet trained")
    end
    if is(rf.learner.oob_error, nothing)
        error("classifier was trained with do_oob=false")
    end

    rf.learner.oob_error
end
