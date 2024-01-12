const std = @import("std");
const assert = std.debug.assert;

pub const Dense = @import("layers/Dense.zig");

pub const LayerTag = enum {
    dense,
    // conv2d,
    // max_pool2d,
    // avg_pool2d,
};

pub const Layer = union(LayerTag) {
    dense: Dense,

    pub fn forward(self: Layer, input: []const f32, output: []f32) void {
        switch (self) {
            .dense => |layer| layer.forward(input, output),
        }
    }

    pub fn outputSize(self: Layer) usize {
        return switch (self) {
            .dense => |layer| layer.outputSize(),
        };
    }
};
