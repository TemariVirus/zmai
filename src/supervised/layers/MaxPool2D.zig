//! An average pooling layer with optional padding that works on arrays of 2D
//! slices. This layer has no trainable parameters. The backpropagation
//! implementation assumes that the pooling layer is non-overlapping (i.e.,
//! `stride.x >= pool_size.x and stride.y >= pool_size.y`).
const std = @import("std");
const assert = std.debug.assert;

const layers = @import("../layers.zig");
const Size2D = layers.Size2D;
const Size3D = layers.Size3D;

const Self = @This();

input_shape: Size3D,
pool_size: Size2D,
stride: Size2D,

pub fn init(input_shape: Size3D, pool_size: Size2D, stride: Size2D) Self {
    assert(input_shape.x >= pool_size.x and input_shape.y >= pool_size.y);
    assert(pool_size.x > 0 and pool_size.y > 0);
    assert(stride.x > 0 and stride.y > 0);
    assert(stride.x >= pool_size.x and stride.y >= pool_size.y);

    return .{
        .input_shape = input_shape,
        .pool_size = pool_size,
        .stride = stride,
    };
}

pub fn inputSize(self: Self) usize {
    return self.input_shape.x * self.input_shape.y * self.input_shape.z;
}

pub fn outputSize(self: Self) usize {
    const shape = self.outputShape();
    return shape.x * shape.y * shape.z;
}

/// Returns the shape of the output of this layer.
pub fn outputShape(self: Self) Size3D {
    // TODO: modify to work with padding
    return .{
        .x = (self.input_shape.x - self.pool_size.x) / self.stride.x + 1,
        .y = (self.input_shape.y - self.pool_size.y) / self.stride.y + 1,
        .z = self.input_shape.z,
    };
}

pub fn size(self: Self) usize {
    _ = self;
    return 0;
}

/// `input` should be ordered in such a way that position (x, y, z) corresponds
/// to index `x + (y * input_size.x) + (z * input_size.x * input_size.y)`.
/// `output` will be ordered similarly after the forward pass.
pub fn forward(self: Self, input: []const f32, output: []f32) void {
    assert(input.len == self.inputSize());
    assert(output.len == self.outputSize());

    // TODO: modify to work with padding
    // Loop through each layer
    var out_i: usize = 0;
    for (0..self.input_shape.z) |z| {
        // Slide the pool across the layer
        var y: usize = 0;
        while (y + self.pool_size.y <= self.input_shape.y) : (y += self.stride.y) {
            var x: usize = 0;
            while (x + self.pool_size.x <= self.input_shape.x) : (x += self.stride.x) {
                // Take the average
                output[out_i] = self.maxPool(x, y, z, input);
                out_i += 1;
            }
        }
    }
}

inline fn maxPool(self: Self, x: usize, y: usize, z: usize, input: []const f32) f32 {
    var max: f32 = -std.math.inf(f32);
    var in_i = z * self.input_shape.x * self.input_shape.y + y * self.input_shape.x + x;
    for (0..self.pool_size.y) |_| {
        for (0..self.pool_size.x) |_| {
            max = @max(max, input[in_i]);
            in_i += 1;
        }
        in_i += self.input_shape.x - self.pool_size.x;
    }
    return max;
}

/// Assumes that the pooling layer is non-overlapping.
pub fn backward(
    self: Self,
    input: []f32,
    output: []f32,
    output_grad: []const f32,
    deltas: []f32,
) void {
    assert(input.len == self.inputSize());
    assert(output_grad.len == self.outputSize());
    assert(deltas.len == self.size());

    var out_i: usize = 0;
    // Loop through each layer
    for (0..self.input_shape.z) |z| {
        // Slide the pool across the layer
        var y: usize = 0;
        while (y + self.pool_size.y <= self.input_shape.y) : (y += self.stride.y) {
            var x: usize = 0;
            while (x + self.pool_size.x <= self.input_shape.x) : (x += self.stride.x) {
                // Take the average
                self.backwardKernelInput(
                    x,
                    y,
                    z,
                    input,
                    output[out_i],
                    output_grad[out_i],
                );
                out_i += 1;
            }
        }
    }
}

inline fn backwardKernelInput(
    self: Self,
    x: usize,
    y: usize,
    z: usize,
    input: []f32,
    output: f32,
    gradient: f32,
) void {
    var in_i = z * self.input_shape.x * self.input_shape.y + y * self.input_shape.x + x;
    for (0..self.pool_size.y) |_| {
        for (0..self.pool_size.x) |_| {
            input[in_i] = if (input[in_i] == output) gradient else 0;
            in_i += 1;
        }
        in_i += self.input_shape.x - self.pool_size.x;
    }
}
