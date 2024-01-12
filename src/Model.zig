const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const root = @import("root.zig");
const Layer = root.layers.Layer;
const Loss = root.losses.Loss;

const Self = @This();

layers: []Layer,

/// Allocates the arrays needed to store the activations of the model.
pub fn initActivations(self: Self, allocator: Allocator) ![][]f32 {
    var activations = try allocator.alloc([]f32, self.layers.len + 1);
    activations[0] = try allocator.alloc(f32, self.layers[0].inputSize());

    for (self.layers, 1..) |layer, i| {
        activations[i] = try allocator.alloc(f32, layer.outputSize());
    }
    return activations;
}

/// Frees the arrays allocated by `initActivations`. The allocator passed in
/// must be the same one used in `initActivations`.
pub fn deinitActivations(allocator: Allocator, activations: [][]f32) void {
    for (activations) |layer| {
        allocator.free(layer);
    }
    allocator.free(activations);
}

/// Allocates the arrays needed to store the deltas of the model.
pub fn initDeltas(self: Self, allocator: Allocator) ![][]f32 {
    var deltas = try allocator.alloc([]f32, self.layers.len);
    for (self.layers, 0..) |layer, i| {
        deltas[i] = try allocator.alloc(f32, layer.size());
    }
    return deltas;
}

/// Frees the arrays allocated by `initDeltas`. The allocator passed in must be
/// the same one used in `initDeltas`.
pub fn deinitDeltas(allocator: Allocator, deltas: [][]f32) void {
    for (deltas) |layer| {
        allocator.free(layer);
    }
    allocator.free(deltas);
}

/// Does a forward pass through the model, and stores the activations in
/// `activations`.
pub fn forward(self: Self, input: []const f32, activations: []const []f32) void {
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
    allocator: Allocator,
    loss: Loss,
    activations: []const []f32,
    deltas: []const []f32,
    y_true: []const f32,
) !f32 {
    assert(activations.len == self.layers.len + 1);
    assert(deltas.len == self.layers.len);
    assert(y_true.len == self.layers[self.layers.len - 1].outputSize());

    const loss_value = loss.forward(activations[self.layers.len], y_true);

    const max_size = blk: {
        var max: usize = 0;
        for (activations) |a| {
            max = @max(max, a.len);
        }
        break :blk max;
    };
    var last_activation = try allocator.alloc(f32, max_size);
    var last_activation2 = try allocator.alloc(f32, max_size);
    defer allocator.free(last_activation);
    defer allocator.free(last_activation2);

    var i = self.layers.len;
    @memcpy(last_activation[0..activations[i].len], activations[i]);
    loss.backward(activations[i], y_true);

    while (i > 0) : (i -= 1) {
        @memcpy(last_activation2[0..activations[i - 1].len], activations[i - 1]);
        self.layers[i - 1].backward(
            activations[i - 1],
            last_activation[0..activations[i].len],
            activations[i],
            deltas[i - 1],
        );

        const temp = last_activation;
        last_activation = last_activation2;
        last_activation2 = temp;
    }

    return loss_value;
}

/// Updates the trainable parameters of the model.
pub fn update(self: Self, deltas: []const []const f32, learning_rate: f32) void {
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
        if (!allocator.resize(output, layer.outputSize())) {
            allocator.free(output);
            output = try allocator.alloc(f32, layer.outputSize());
        }
        layer.forward(input2, output);
    }

    allocator.free(input2);
    return output;
}
