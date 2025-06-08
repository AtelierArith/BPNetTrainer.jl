struct LayerAcceptsInputWithDataAndLabels{L} <: LuxCore.AbstractLuxWrapperLayer{:layers}
    layers::L
    function LayerAcceptsInputWithDataAndLabels(layers...)
        l = Parallel(+, layers...)
        new{typeof(l)}(l)
    end
end

function (m::LayerAcceptsInputWithDataAndLabels)(x, ps, st::NamedTuple)
    y, st = Lux.apply(m.layers, Tuple(x.data), ps, st)
    y * x.labels, st
end

