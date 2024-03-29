//! A dense (i.e., fully connected) layer with a fixed activation function, and
//! trainable weights and biases.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Activation = @import("../activations.zig").Activation;

const Self = @This();

/// The weights of the layer. The nth `inputSize()` elements are the weights
/// of the connections between the nth output neuron and the input neurons.
weights: []f32,
biases: []f32,
activation: Activation,

/// Initializes a dense layer. The initial parameters are generated by calling
/// `randomF32`.
pub fn init(
    allocator: Allocator,
    input_size: usize,
    output_size: usize,
    activation: Activation,
    comptime randomF32: fn () f32,
) !Self {
    // Normalisation needed for numerical stability
    const normalisation = 1.0 / @as(f32, @floatFromInt(input_size));

    const weights = try allocator.alloc(f32, output_size * input_size);
    for (weights) |*w| {
        w.* = randomF32() * normalisation;
    }

    const biases = try allocator.alloc(f32, output_size);
    for (biases) |*b| {
        b.* = randomF32();
    }

    return .{
        .weights = weights,
        .biases = biases,
        .activation = activation,
    };
}

/// Frees the memory used by this layer. `allocator` is the same allocator that
/// was passed in to `init`.
pub fn deinit(self: Self, allocator: Allocator) void {
    allocator.free(self.weights);
    allocator.free(self.biases);
}

pub fn inputSize(self: Self) usize {
    return self.weights.len / self.biases.len;
}

pub fn outputSize(self: Self) usize {
    return self.biases.len;
}

pub fn size(self: Self) usize {
    return self.weights.len + self.biases.len;
}

pub fn forward(self: Self, input: []const f32, output: []f32) void {
    assert(input.len == self.inputSize());
    assert(output.len == self.outputSize());

    @memcpy(output, self.biases);

    var k: usize = 0;
    for (0..output.len) |i| {
        for (0..input.len) |j| {
            output[i] += input[j] * self.weights[k];
            k += 1;
        }
    }

    self.activation.forward(output);
}

pub fn backward(
    self: Self,
    input: []f32,
    output: []f32,
    output_grad: []const f32,
    deltas: []f32,
) void {
    assert(input.len == self.inputSize());
    assert(output.len == self.outputSize());
    assert(output_grad.len == self.outputSize());
    assert(deltas.len == self.size());

    self.activation.backward(output);

    var k: usize = 0;
    for (0..output.len) |i| {
        const grad = output[i] * output_grad[i];
        deltas[self.weights.len + i] -= grad;
        for (0..input.len) |j| {
            deltas[k] -= grad * input[j];
            k += 1;
        }
    }

    k = 0;
    for (input) |*value| {
        value.* = 0.0;
    }
    for (0..output.len) |i| {
        const grad = output[i] * output_grad[i];
        for (0..input.len) |j| {
            input[j] += grad * self.weights[k];
            k += 1;
        }
    }
}

pub fn update(self: Self, deltas: []const f32, learning_rate: f32) void {
    assert(deltas.len == self.size());

    for (0..self.weights.len) |i| {
        self.weights[i] += learning_rate * deltas[i];
    }
    for (0..self.biases.len) |i| {
        self.biases[i] += learning_rate * deltas[self.weights.len + i];
    }
}
