//! Implements Stochastic Gradient Descent (SGD) for training a neural network.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Batcher = @import("../optimizers.zig").Batcher;
const Loss = @import("../losses.zig").Loss;
const Model = @import("../Model.zig");

const Self = @This();

allocator: Allocator,
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
        .allocator = allocator,
        .model = model,
        .activations = activations,
        .aux_activations = [_][]f32{
            try allocator.alloc(f32, max_size),
            try allocator.alloc(f32, max_size),
        },
        .deltas = deltas,
    };
}

pub fn deinit(self: Self) void {
    for (self.activations) |arr| {
        self.allocator.free(arr);
    }
    self.allocator.free(self.activations);

    for (self.aux_activations) |arr| {
        self.allocator.free(arr);
    }

    for (self.deltas) |arr| {
        self.allocator.free(arr);
    }
    self.allocator.free(self.deltas);
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
) !void {
    assert(x_data.len == y_data.len);

    var batcher = try Batcher.init(self.allocator, batch_size, x_data, y_data);
    defer batcher.deinit(self.allocator);

    for (0..epochs) |epoch| {
        var loss: f32 = 0.0;

        while (batcher.next()) |batch| {
            loss += self.fitOnce(
                batch.x,
                batch.y,
                loss_fn,
                learning_rate,
            );
        }

        loss /= @floatFromInt(x_data.len);
        std.debug.print("Epoch {d}, loss: {d}\n", .{ epoch + 1, loss });
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

    self.model.update(
        self.deltas,
        learning_rate / @as(f32, @floatFromInt(x_data.len)),
    );
    return loss;
}
