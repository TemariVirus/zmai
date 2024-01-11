const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Activation = @import("../activations.zig").Activation;
const Layer = @import("../layers.zig").Layer;

const Self = @This();

weights: []f32,
biases: []f32,
activation: Activation,

pub fn init(allocator: Allocator, input_size: usize, output_size: usize, activation: Activation, rand_fn: *const fn () f32) !Self {
    const weights = try allocator.alloc(f32, output_size * input_size);
    for (weights) |*w| {
        w.* = rand_fn();
    }

    const biases = try allocator.alloc(f32, output_size);
    for (biases) |*b| {
        b.* = rand_fn();
    }

    return .{
        .weights = weights,
        .biases = biases,
        .activation = activation,
    };
}

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

pub fn forward(self: *Self, input: []const f32, output: []f32) void {
    assert(input.len == self.inputSize());
    assert(output.len == self.outputSize());

    @memcpy(output, self.biases);

    for (0..self.outputSize()) |i| {
        const row_start = i * self.inputSize();
        for (0..self.inputSize()) |j| {
            output[i] += input[j] * self.weights[row_start + j];
        }
    }

    self.activation(output);
}

pub fn layer(self: *Self) Layer {
    return Layer.init(self, forward, self.inputSize(), self.outputSize());
}
