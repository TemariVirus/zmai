const std = @import("std");
const Allocator = std.mem.Allocator;

const root = @import("root.zig");
const Dense = root.layers.Dense;
const Layer = root.layers.Layer;
const Model = root.Model;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var dense1 = try Dense.init(allocator, 2, 2, .relu, gaussianRandom);
    var dense2 = try Dense.init(allocator, 2, 1, .sigmoid, gaussianRandom);
    defer dense1.deinit(allocator);
    defer dense2.deinit(allocator);

    // XOR gate
    const xor_x = [_][]const f32{
        &.{ 0, 0 },
        &.{ 0, 1 },
        &.{ 1, 0 },
        &.{ 1, 1 },
    };
    const xor_y = [_][]const f32{
        &.{0},
        &.{1},
        &.{1},
        &.{0},
    };

    var layers = [_]Layer{
        .{ .dense = dense1 },
        .{ .dense = dense2 },
    };
    const model = Model{
        .layers = &layers,
    };

    for (xor_x) |x| {
        const y = try model.predict(allocator, x);
        defer allocator.free(y);
        std.debug.print("{d} ^ {d} -> {d}\n", .{ x[0], x[1], y[0] });
    }

    std.debug.print("\n", .{});
    try fit(allocator, model, &xor_x, &xor_y, 1.0, 500);

    for (xor_x) |x| {
        const y = try model.predict(allocator, x);
        defer allocator.free(y);
        std.debug.print("{d} ^ {d} -> {d}\n", .{ x[0], x[1], y[0] });
    }
}

fn gaussianRandom() f32 {
    const State = struct {
        var rand = std.rand.DefaultPrng.init(23);
        const random = rand.random();
    };
    return State.random.floatNorm(f32);
}

fn fit(
    allocator: Allocator,
    model: Model,
    x_data: []const []const f32,
    y_data: []const []const f32,
    learning_rate: f32,
    epochs: usize,
) !void {
    const activations = try model.initActivations(allocator);
    const deltas = try model.initDeltas(allocator);
    defer Model.deinitActivations(allocator, activations);
    defer Model.deinitDeltas(allocator, deltas);

    for (0..epochs) |_| {
        var loss: f32 = 0.0;
        for (deltas) |row| {
            for (row) |*value| {
                value.* = 0.0;
            }
        }

        for (x_data, y_data) |x, y| {
            model.forward(x, activations);
            loss += try model.backward(
                allocator,
                .mean_squared_error,
                activations,
                deltas,
                y,
            );
        }

        loss /= @floatFromInt(x_data.len);
        // std.debug.print("{}\n", .{loss});
        model.update(deltas, learning_rate);
    }
}
