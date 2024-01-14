const std = @import("std");

const zmai = @import("zmai");
const supervised = zmai.supervised;
const Dense = supervised.layers.Dense;
const Layer = supervised.layers.Layer;
const Model = supervised.Model;
const Sgd = supervised.optimizers.Sgd;

// XOR truth table
const x = [_][]const f32{
    &.{ 0, 0 },
    &.{ 0, 1 },
    &.{ 1, 0 },
    &.{ 1, 1 },
};
const y = [_][]const f32{
    &.{0},
    &.{1},
    &.{1},
    &.{0},
};

// Trains a simple model to learn the binary XOR function.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Define the model
    zmai.setRandomSeed(23);
    var dense1 = try Dense.init(allocator, 2, 2, .relu, zmai.gaussianRandom);
    var dense2 = try Dense.init(allocator, 2, 1, .sigmoid, zmai.gaussianRandom);
    defer dense1.deinit(allocator);
    defer dense2.deinit(allocator);

    var layers = [_]Layer{
        .{ .dense = dense1 },
        .{ .dense = dense2 },
    };
    const model = Model{
        .layers = &layers,
    };

    // Create stochastic gradient descent optimiser and train the model
    const sgd = try Sgd.init(allocator, model);
    try sgd.fit(
        &x,
        &y,
        10_000,
        x.len,
        .mean_squared_error,
        2.5,
    );
    sgd.deinit();

    // Print the model's predictions
    std.debug.print("\n", .{});
    for (x) |in| {
        const out = try model.predict(allocator, in);
        defer allocator.free(out);
        std.debug.print("{d} ^ {d} -> {d}\n", .{ in[0], in[1], out[0] });
    }
}
