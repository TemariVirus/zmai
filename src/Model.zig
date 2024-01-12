const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const root = @import("root.zig");
const Layer = @import("layers.zig").Layer;

const Self = @This();

layers: []Layer,

pub fn forward(self: Self, allocator: Allocator, input: []const f32) ![][]f32 {
    assert(self.layers.len > 0);

    const activations = try allocator.alloc([]f32, self.layers.len);
    activations[0] = try allocator.alloc(f32, self.layers[0].output_size());
    self.layers[0].forward(input, activations[0]);

    for (self.layers[1..], 1..) |layer, i| {
        activations[i] = try allocator.alloc(f32, layer.output_size());
        layer.forward(activations[i - 1], activations[i]);
    }

    return activations;
}

pub fn predict(self: Self, allocator: Allocator, input: []const f32) ![]f32 {
    assert(self.layers.len > 0);

    var input2 = try allocator.alloc(f32, self.layers[0].outputSize());
    self.layers[0].forward(input, input2);
    if (self.layers.len == 1) {
        return input2;
    }

    var output = try allocator.alloc(f32, self.layers[1].outputSize());
    self.layers[1].forward(input2, output);
    for (self.layers[2..]) |layer| {
        const temp = input2;
        input2 = output;
        output = temp;

        // If resize fails, fall back to freeing and allocating
        if (!allocator.resize(output, layer.outputSize())) {
            allocator.free(output);
            output = try allocator.alloc(f32, layer.outputSize());
        }
        layer.forward(input2, output);
    }

    allocator.free(input2);
    return output;
}
