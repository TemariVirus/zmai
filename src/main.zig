const std = @import("std");

const root = @import("root.zig");
const Dense = root.layers.Dense;
const activations = root.activations;
const Model = root.Model;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var layer1 = try Dense.init(allocator, 3, 2, activations.relu, gaussianRandom);
    defer layer1.deinit(allocator);

    const input = [_]f32{ -3.0, 2.1, 1.0 };

    var layers = [_]root.layers.Layer{layer1.layer()};
    const model = Model{
        .layers = &layers,
    };

    const output = try model.forward(allocator, &input);
    defer allocator.free(output);

    std.debug.print("{any}\n", .{layer1});
    std.debug.print("{any}\n", .{output});
}

fn gaussianRandom() f32 {
    const State = struct {
        var rand = std.rand.DefaultPrng.init(0);
        const random = rand.random();
    };
    return State.random.floatNorm(f32);
}
