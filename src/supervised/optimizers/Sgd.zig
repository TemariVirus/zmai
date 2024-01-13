//! Implements Stochastic Gradient Descent (SGD) for training a neural network.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Loss = @import("../losses.zig").Loss;
const Model = @import("../Model.zig");

const Self = @This();

model: Model,
activations: [][]f32,
aux_activations: [2][]f32,
deltas: [][]f32,

pub fn init(allocator: Allocator, model: Model) !Self {
    const activations = try allocator.alloc([]f32, model.layers.len + 1);
    activations[0] = try allocator.alloc(f32, model.layers[0].inputSize());

    var max_size = activations[0].len;
    for (model.layers, 1..) |layer, i| {
        activations[i] = try allocator.alloc(f32, layer.outputSize());
        max_size = @max(max_size, activations[i].len);
    }

    const deltas = try allocator.alloc([]f32, model.layers.len);
    for (model.layers, 0..) |layer, i| {
        deltas[i] = try allocator.alloc(f32, layer.size());
    }

    return Self{
        .model = model,
        .activations = activations,
        .aux_activations = [_][]f32{ try allocator.alloc(f32, max_size), try allocator.alloc(f32, max_size) },
        .deltas = deltas,
    };
}

/// `allocator` must be the same allocator used in `init`.
pub fn deinit(self: Self, allocator: Allocator) void {
    for (self.activations) |arr| {
        allocator.free(arr);
    }
    allocator.free(self.activations);

    for (self.aux_activations) |arr| {
        allocator.free(arr);
    }

    for (self.deltas) |arr| {
        allocator.free(arr);
    }
    allocator.free(self.deltas);
}

// TODO: Add callbacks for logging, saving, etc.
pub fn fit(
    self: Self,
    x_data: []const []const f32,
    y_data: []const []const f32,
    epochs: usize,
    batch_size: usize,
    loss_fn: Loss,
    learning_rate: f32,
) void {
    assert(x_data.len == y_data.len);
    assert(batch_size > 0 and batch_size <= x_data.len);

    // TODO: shuffle data
    const batch_count = std.math.divCeil(usize, x_data.len, batch_size) catch unreachable;
    for (0..epochs) |i| {
        var loss: f32 = 0.0;

        for (0..batch_count) |j| {
            const start = j * batch_size;
            const end = @min(start + batch_size, x_data.len);
            loss += self.fitOnce(
                x_data[start..end],
                y_data[start..end],
                loss_fn,
                learning_rate,
            );
        }

        loss /= @floatFromInt(x_data.len);
        std.debug.print("Epoch {d}, loss: {d}\n", .{ i + 1, loss });
    }
}

pub fn fitOnce(
    self: Self,
    x_data: []const []const f32,
    y_data: []const []const f32,
    loss_fn: Loss,
    learning_rate: f32,
) f32 {
    var loss: f32 = 0.0;
    for (self.deltas) |row| {
        for (row) |*value| {
            value.* = 0.0;
        }
    }

    for (x_data, y_data) |x, y| {
        self.model.forward(x, self.activations);
        loss += self.model.backward(
            loss_fn,
            self.activations,
            self.aux_activations,
            self.deltas,
            y,
        );
    }

    self.model.update(self.deltas, learning_rate / @as(f32, @floatFromInt(x_data.len)));
    return loss;
}
