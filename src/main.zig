const std = @import("std");

const root = @import("root.zig");
const Dense = root.layers.Dense;
const Layer = root.layers.Layer;
const Model = root.Model;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var dense1 = try Dense.init(allocator, 3, 3, .leaky_relu, gaussianRandom);
    var dense2 = try Dense.init(allocator, 3, 2, .leaky_relu, gaussianRandom);
    var dense3 = try Dense.init(allocator, 2, 10, .sigmoid, gaussianRandom);

    defer dense1.deinit(allocator);
    defer dense2.deinit(allocator);
    defer dense3.deinit(allocator);

    const input = [_]f32{ -3.0, 2.1, 1.0 };

    var layers = [_]Layer{
        .{ .dense = dense1 },
        .{ .dense = dense2 },
        .{ .dense = dense3 },
    };
    const model = Model{
        .layers = &layers,
    };

    const output = try model.predict(allocator, &input);
    defer allocator.free(output);

    std.debug.print("{any}\n", .{output});
}

fn gaussianRandom() f32 {
    const State = struct {
        var rand = std.rand.DefaultPrng.init(0);
        const random = rand.random();
    };
    return State.random.floatNorm(f32);
}
