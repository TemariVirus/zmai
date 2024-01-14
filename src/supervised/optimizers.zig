const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const random = @import("../root.zig").random;

pub const Sgd = @import("optimizers/Sgd.zig");

// TODO
// pub const Adam = @import("optimizers/Adam.zig");

/// Iterates over the data in batches of a specified size, shuffling the data
/// every epoch.
pub const Batcher = struct {
    batch_size: usize,
    current: usize,
    indices: []usize,
    x_data: []const []const f32,
    y_data: []const []const f32,
    batch_x: [][]const f32,
    batch_y: [][]const f32,

    pub fn init(
        allocator: Allocator,
        batch_size: usize,
        x_data: []const []const f32,
        y_data: []const []const f32,
    ) !Batcher {
        assert(batch_size > 0 and batch_size <= x_data.len);

        const indices = try allocator.alloc(usize, x_data.len);
        for (0..indices.len) |i| {
            indices[i] = i;
        }

        return Batcher{
            .batch_size = batch_size,
            .current = 0,
            .indices = indices,
            .x_data = x_data,
            .y_data = y_data,
            .batch_x = try allocator.alloc([]const f32, batch_size),
            .batch_y = try allocator.alloc([]const f32, batch_size),
        };
    }

    /// The allocator passed in must be the same allocator used to allocate the
    /// Batcher.
    pub fn deinit(self: Batcher, allocator: Allocator) void {
        allocator.free(self.indices);
        allocator.free(self.batch_x);
        allocator.free(self.batch_y);
    }

    pub fn next(self: *Batcher) ?struct { x: []const []const f32, y: []const []const f32 } {
        if (self.current >= self.indices.len) {
            self.current = 0;
            return null;
        }
        if (self.current == 0) {
            random.shuffle(usize, self.indices);
        }

        const start = self.current;
        for (0..self.batch_size) |i| {
            if (self.current == self.indices.len) {
                break;
            }

            const index = self.indices[self.current];
            self.current += 1;

            self.batch_x[i] = self.x_data[index];
            self.batch_y[i] = self.y_data[index];
        }

        const len = self.current - start;
        return .{
            .x = self.batch_x[0..len],
            .y = self.batch_y[0..len],
        };
    }
};
