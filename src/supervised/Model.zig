const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Layer = @import("layers.zig").Layer;
const Loss = @import("losses.zig").Loss;

const Self = @This();

layers: []const Layer,

/// The number of trainable parameters in the model.
pub fn size(self: Self) usize {
    var count: usize = 0;
    for (self.layers) |layer| {
        count += layer.size();
    }
    return count;
}

/// Does a forward pass through the model, and stores the activations in
/// `activations`.
pub fn forward(
    self: Self,
    input: []const f32,
    activations: []const []f32,
) void {
    assert(activations.len == self.layers.len + 1);

    @memcpy(activations[0], input);
    for (self.layers, 1..) |layer, i| {
        layer.forward(activations[i - 1], activations[i]);
    }
}

/// Does a backward pass through the model, and returns the value of the loss
/// function.
pub fn backward(
    self: Self,
    loss_fn: Loss,
    activations: [][]f32,
    aux_activations: [2][]f32,
    deltas: []const []f32,
    y_true: []const f32,
) f32 {
    assert(activations.len == self.layers.len + 1);
    assert(deltas.len == self.layers.len);
    assert(y_true.len == self.layers[self.layers.len - 1].outputSize());

    const loss = loss_fn.forward(activations[self.layers.len], y_true);
    var current = aux_activations[0];
    var previous = aux_activations[1];

    var i = self.layers.len;
    @memcpy(current[0..activations[i].len], activations[i]);
    loss_fn.backward(activations[i], y_true);

    while (i > 0) : (i -= 1) {
        @memcpy(previous[0..activations[i - 1].len], activations[i - 1]);
        self.layers[i - 1].backward(
            activations[i - 1],
            current[0..activations[i].len],
            activations[i],
            deltas[i - 1],
        );

        const temp = current;
        current = previous;
        previous = temp;
    }

    return loss;
}

/// Updates the trainable parameters of the model.
pub fn update(
    self: Self,
    deltas: []const []const f32,
    learning_rate: f32,
) void {
    assert(deltas.len == self.layers.len);

    for (self.layers, deltas) |layer, delta| {
        layer.update(delta, learning_rate);
    }
}

/// Returns the output of the model for the given input.
pub fn predict(self: Self, allocator: Allocator, input: []const f32) ![]f32 {
    assert(self.layers.len > 0);

    var input2 = try allocator.alloc(f32, self.layers[0].outputSize());
    self.layers[0].forward(input, input2);
    if (self.layers.len == 1) {
        return input2;
    }

    var output = try allocator.alloc(f32, self.layers[1].outputSize());
    self.layers[1].forward(input2, output);
    for (self.layers[2..]) |layer| {
        const temp = input2;
        input2 = output;
        output = temp;

        // If resize fails, fall back to freeing and allocating
        if (output.len < layer.outputSize() and
            !allocator.resize(output, layer.outputSize()))
        {
            allocator.free(output);
            output = try allocator.alloc(f32, layer.outputSize());
        }
        layer.forward(input2[0..layer.inputSize()], output[0..layer.outputSize()]);
    }
    allocator.free(input2);

    // Output may be too large, size it properly
    const out_size = self.layers[self.layers.len - 1].outputSize();
    if (output.len == out_size) {
        return output;
    }

    const temp = output;
    defer allocator.free(temp);

    output = try allocator.alloc(f32, out_size);
    @memcpy(output, temp[0..out_size]);
    return output;
}
