//! A 2D convolutional layer, with a fixed kernel size, stride, padding,
//! activation function, and trainable weights and biases.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Activation = @import("../activations.zig").Activation;
const layers = @import("../layers.zig");
const Size2D = layers.Size2D;
const Size3D = layers.Size3D;

// TODO: try out intel MKL for performance improvements
const Self = @This();

weights: []f32,
biases: []f32,
input_size: Size2D,
kernel_size: Size2D,
stride: Size2D,
activation: Activation,

/// Initializes a dense layer. The initial parameters are generated by calling
/// `randomF32`.
pub fn init(
    allocator: Allocator,
    input_size: Size3D,
    output_channels: usize,
    kernel_size: Size2D,
    stride: Size2D,
    activation: Activation,
    comptime randomF32: fn () f32,
) !Self {
    assert(input_size.z > 0);
    assert(input_size.x >= kernel_size.x and input_size.y >= kernel_size.y);
    assert(kernel_size.x > 0 and kernel_size.y > 0);
    assert(stride.x > 0 and stride.y > 0);

    // Normalisation needed for numerical stability
    const kernel_volume = kernel_size.x * kernel_size.y * input_size.z;
    const normalisation = 1.0 / @as(f32, @floatFromInt(kernel_volume));
    const weights = try allocator.alloc(
        f32,
        output_channels * kernel_volume,
    );
    for (weights) |*w| {
        w.* = randomF32() * normalisation;
    }

    const biases = try allocator.alloc(f32, output_channels);
    for (biases) |*b| {
        b.* = randomF32();
    }

    return .{
        .weights = weights,
        .biases = biases,
        .input_size = .{
            .x = input_size.x,
            .y = input_size.y,
        },
        .kernel_size = kernel_size,
        .stride = stride,
        .activation = activation,
    };
}

/// Frees the memory used by this layer. `allocator` is the same allocator that
/// was passed in to `init`.
pub fn deinit(self: Self, allocator: Allocator) void {
    allocator.free(self.weights);
    allocator.free(self.biases);
}

/// Returns the number of channels in the input of this layer.
pub fn inputChannels(self: Self) usize {
    return self.weights.len / (self.kernel_size.x * self.kernel_size.y) / self.channels();
}

/// Returns the number of channels in the output of this layer.
pub fn channels(self: Self) usize {
    return self.biases.len;
}

pub fn inputSize(self: Self) usize {
    return self.input_size.x * self.input_size.y * self.inputChannels();
}

pub fn outputSize(self: Self) usize {
    const shape = self.outputShape();
    return shape.x * shape.y * shape.z;
}

/// Returns the shape of the output of this layer.
pub fn outputShape(self: Self) Size3D {
    // TODO: modify to work with padding
    return .{
        .x = (self.input_size.x - self.kernel_size.x) / self.stride.x + 1,
        .y = (self.input_size.y - self.kernel_size.y) / self.stride.y + 1,
        .z = self.channels(),
    };
}

pub fn size(self: Self) usize {
    return self.weights.len + self.biases.len;
}

/// `input` should be ordered in such a way that position (x, y, z) corresponds
/// to index `x + (y * input_size.x) + (z * input_size.x * input_size.y)`.
/// `output` will be ordered similarly after the forward pass.
// TODO: See if changing the order of loops increases performance for forward and backward passes
pub fn forward(self: Self, input: []const f32, output: []f32) void {
    assert(input.len == self.inputSize());
    assert(output.len == self.outputSize());

    // TODO: modify to work with padding
    // Loop through each kernel
    var out_i: usize = 0;
    for (0..self.channels()) |kernel_i| {
        // Slide the kernel across the input
        var y: usize = 0;
        while (y + self.kernel_size.y <= self.input_size.y) : (y += self.stride.y) {
            var x: usize = 0;
            while (x + self.kernel_size.x <= self.input_size.x) : (x += self.stride.x) {
                // Multiply the kernel with the input and add to the output
                output[out_i] = self.forwardKernel(kernel_i, x, y, input);
                out_i += 1;
            }
        }
    }

    self.activation.forward(output);
}

inline fn forwardKernel(self: Self, kernel_i: usize, x: usize, y: usize, input: []const f32) f32 {
    var sum: f32 = self.biases[kernel_i];

    var weight_i = kernel_i * self.kernel_size.x * self.kernel_size.y * self.inputChannels();
    for (0..self.inputChannels()) |z| {
        var in_i = z * self.input_size.x * self.input_size.y + y * self.input_size.x + x;
        for (0..self.kernel_size.y) |_| {
            for (0..self.kernel_size.x) |_| {
                sum += input[in_i] * self.weights[weight_i];
                in_i += 1;
                weight_i += 1;
            }
            in_i += self.input_size.x - self.kernel_size.x;
        }
    }

    return sum;
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

    // TODO: modify to work with padding
    // Loop through each kernel
    var out_i: usize = 0;
    for (0..self.channels()) |kernel_i| {
        // Slide the kernel across the input
        var y: usize = 0;
        while (y + self.kernel_size.y <= self.input_size.y) : (y += self.stride.y) {
            var x: usize = 0;
            while (x + self.kernel_size.x <= self.input_size.x) : (x += self.stride.x) {
                // Multiply the kernel with the input and add to the output
                self.backwardKernelDeltas(
                    kernel_i,
                    x,
                    y,
                    input,
                    deltas,
                    output[out_i] * output_grad[out_i],
                );
                out_i += 1;
            }
        }
    }

    out_i = 0;
    for (input) |*value| {
        value.* = 0.0;
    }
    for (0..self.channels()) |kernel_i| {
        // Slide the kernel across the input
        var y: usize = 0;
        while (y + self.kernel_size.y <= self.input_size.y) : (y += self.stride.y) {
            var x: usize = 0;
            while (x + self.kernel_size.x <= self.input_size.x) : (x += self.stride.x) {
                // Multiply the kernel with the input and add to the output
                self.backwardKernelInput(
                    kernel_i,
                    x,
                    y,
                    input,
                    output[out_i] * output_grad[out_i],
                );
                out_i += 1;
            }
        }
    }
}

inline fn backwardKernelDeltas(self: Self, kernel_i: usize, x: usize, y: usize, input: []const f32, deltas: []f32, gradient: f32) void {
    deltas[self.weights.len + kernel_i] -= gradient;

    var weight_i = kernel_i * self.kernel_size.x * self.kernel_size.y * self.inputChannels();
    for (0..self.inputChannels()) |z| {
        var in_i = z * self.input_size.x * self.input_size.y + y * self.input_size.x + x;
        for (0..self.kernel_size.y) |_| {
            for (0..self.kernel_size.x) |_| {
                deltas[weight_i] -= gradient * input[in_i];
                in_i += 1;
                weight_i += 1;
            }
            in_i += self.input_size.x - self.kernel_size.x;
        }
    }
}

inline fn backwardKernelInput(self: Self, kernel_i: usize, x: usize, y: usize, input: []f32, gradient: f32) void {
    var weight_i = kernel_i * self.kernel_size.x * self.kernel_size.y * self.inputChannels();
    for (0..self.inputChannels()) |z| {
        var in_i = z * self.input_size.x * self.input_size.y + y * self.input_size.x + x;
        for (0..self.kernel_size.y) |_| {
            for (0..self.kernel_size.x) |_| {
                input[in_i] += gradient * self.weights[weight_i];
                in_i += 1;
                weight_i += 1;
            }
            in_i += self.input_size.x - self.kernel_size.x;
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
