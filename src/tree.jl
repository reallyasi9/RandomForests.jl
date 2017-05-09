module Trees

using StatsBase
using DataFrames

include("example.jl")
include("sort.jl")
include("split.jl")

export Tree, fit!, predict

abstract Element

type Node{T} <: Element
    feature::Int
    threshold::T
    impurity::Float64
    n_samples::Int
    left::Element
    right::Element
    depth::Int
end

Base.show(io::Base.IO, node::Node) = print(io, "Node{depth: ", node.depth, ", feature[", node.feature, "] <= ", node.threshold, ", splits ", node.n_samples, " samples with impurity ", node.impurity, "}")

abstract Leaf <: Element

type ClassificationLeaf <: Leaf
    counts::Vector{Int}
    impurity::Float64
    n_samples::Int
    depth::Int

    function ClassificationLeaf(example::Example, samples::Vector{Int}, impurity::Float64, depth::Int)
        counts = zeros(Int, example.n_labels)
        for s in samples
            label = example.y[s]
            counts[label] += round(Int, example.sample_weight[s])
        end
        new(counts, impurity, length(samples), depth)
    end
end

Base.show(io::Base.IO, node::ClassificationLeaf) = print(io, "ClassificationLeaf{depth: ", node.depth, ", contains ", node.n_samples, " with impurity ", node.impurity, " and majority ",  majority(node), "}")

majority(leaf::ClassificationLeaf) = indmax(leaf.counts)

type RegressionLeaf <: Leaf
    mean::Float64
    impurity::Float64
    n_samples::Int
    depth::Int

    function RegressionLeaf(example::Example, samples::Vector{Int}, impurity::Float64, depth::Int)
        new(mean(example.y[samples]), impurity, length(samples), depth)
    end
end

Base.show(io::Base.IO, node::RegressionLeaf) = print(io, "RegressionLeaf{depth: ", node.depth, ", contains ", node.n_samples, " with impurity ", node.impurity, " and mean ",  mean(node), "}")

Base.mean(leaf::RegressionLeaf) = leaf.mean

immutable Undef <: Element; end
const undef = Undef()

"Parameters to build a tree"
immutable Params
    criterion::Criterion
    splitter  # splitter constructor
    max_features::Int
    max_depth::Int
    min_samples_split::Int
end

"Bundled arguments for splitting a node"
immutable SplitArgs
    depth::Int
    range::UnitRange{Int}
end

type Tree
    root::Element
    sample_weights::Vector{Float64}
    Tree() = new(undef, Float64[])
end

Base.show(io::Base.IO, tree::Tree) = print(io, "Tree{", nnodes(tree), " elements and ", nleaves(tree), " leaves, height: ", height(tree), "}")

root(tree::Tree) = tree.root
nnodes(tree::Tree) = sum(1 for x in DFSIterator(tree))
sampleweights(tree::Tree) = tree.sample_weights

nleaves(tree::Tree) = sum(1 for x in LeafIterator(tree))
height(tree::Tree) = tree.root == undef ? 0 : mapreduce(x -> x.depth, max, LeafIterator(tree))

impurity(node::Node) = node.impurity
impurity(leaf::Leaf) = leaf.impurity
nsamples(node::Node) = node.n_samples
nsamples(leaf::Leaf) = leaf.n_samples

"Iterator interface to Tree--traverses leaves in depth-first order (left-to-right)."
type LeafIterator
    tree::Tree
end

function Base.start(li::LeafIterator)
    if root(li.tree) == undef
        return Vector{Element}()
    end
    return Vector{Element}([root(li.tree)])
end

function Base.next(li::LeafIterator, queue::Vector{Element})
    e = undef
    while !isa(e, Leaf)
        e = pop!(queue)
        push_children!(queue, e)
    end
    e, queue
end

Base.done(li::LeafIterator, queue::Vector{Element}) = isempty(queue)

"Iterator interface to Tree--traverses nodes in depth-first order."
type DFSIterator
    tree::Tree
end

function Base.start(ni::DFSIterator)
    if root(ni.tree) == undef
        return Vector{Element}()
    end
    return Vector{Element}([root(ni.tree)])
end

function Base.next(ni::DFSIterator, queue::Vector{Element})
    e = pop!(queue)
    push_children!(queue, e)
    e, queue
end

Base.done(ni::DFSIterator, queue::Vector{Element}) = isempty(queue)

function push_children!(queue::Vector{Element}, e::Leaf)
end
function push_children!(queue::Vector{Element}, e::Node)
    push!(queue, e.right)
    push!(queue, e.left)
end


function fit!(tree::Tree, example::Example, criterion::Criterion, max_features::Int, max_depth::Int, min_samples_split::Int)
    if isa(criterion, ClassificationCriterion)
        splitter = ClassificationSplitter{typeof(criterion)}
    elseif isa(criterion, RegressionCriterion)
        splitter = RegressionSplitter{typeof(criterion)}
    else
        error("invalid criterion")
    end
    params = Params(criterion, splitter, max_features, max_depth, min_samples_split)
    samples = where(example.sample_weight)
    sample_range = 1:length(samples)
    args = SplitArgs(1, sample_range)
    build_tree!(tree, example, samples, args, params)
    return
end


function where(v::AbstractVector)
    n = countnz(v)
    indices = Array(Int, n)
    i = 1
    j = 0

    while (j = findnext(v, j + 1)) > 0
        indices[i] = j
        i += 1
    end

    indices
end


function leaf(example::Example, samples, criterion::RegressionCriterion, depth::Int)
    RegressionLeaf(example, samples, impurity(samples, example, criterion), depth)
end


function leaf(example::Example, samples, criterion::ClassificationCriterion, depth::Int)
    ClassificationLeaf(example, samples, impurity(samples, example, criterion), depth)
end


"""
    build_subtree(example, samples, args, params)

Builds and returns an Element that either splits samples using args (a Node), or contains all samples (a Leaf).
"""
function build_subtree(example::Example, samples::Vector{Int}, args::SplitArgs, params::Params)
    n_features = example.n_features
    range = args.range  # shortcut
    n_samples = length(range)

    if args.depth >= params.max_depth || n_samples < params.min_samples_split
        return leaf(example, samples[range], params.criterion, args.depth)
    end

    best_feature = 0
    best_impurity = Inf
    samp = samples[range]
    # Why local?
    local best_threshold, best_boundary

    # This is the CART part.
    # TODO: replace with a method to calculate splits in other ways.
    for k in sample(1:n_features, params.max_features, replace=false)
        feature = example.x[samp, k]
        sort!(samples, feature, range)
        splitter = params.splitter(samples, feature, range, example)

        for s in splitter
            averaged_impurity = (s.left_impurity * s.n_left_samples + s.right_impurity * s.n_right_samples) / (s.n_left_samples + s.n_right_samples)

            if averaged_impurity < best_impurity
                best_impurity = averaged_impurity
                best_feature = k
                best_threshold = s.threshold
                best_boundary = s.boundary
            end
        end
    end

    if best_feature == 0
        # No best: return a leaf with all samples
        return leaf(example, samples[range], params.criterion, args.depth)
    end

    feature = example.x[samples[range], best_feature]
    sort!(samples, feature, range)

    left = undef
    right = undef

    next_depth = args.depth + 1
    left_node = SplitArgs(next_depth, range[1:best_boundary])
    right_node = SplitArgs(next_depth, range[best_boundary+1:end])
    left = build_subtree(example, samples, left_node, params)
    right = build_subtree(example, samples, right_node, params)
    return Node(best_feature, best_threshold, best_impurity, n_samples, left, right, args.depth)
end


function build_tree!(tree::Tree, example::Example, samples::Vector{Int}, args::SplitArgs, params::Params)
    tree.root = build_subtree(example, samples, args, params)
end


function predict(tree::Tree, x::AbstractVector)
    node = getroot(tree)
    return predict(node, x)
end

function predict(node::Node, x::AbstractVector)
    if x[node.feature] <= node.threshold
        # go left
        return predict(node.left, x)
    else
        # go right
        return predict(node.right, x)
    end
end

predict(leaf::RegressionLeaf, x::AbstractVector) = mean(leaf)
predict(leaf::ClassificationLeaf, x::AbstractVector) = majority(leaf)

end  # module Trees
