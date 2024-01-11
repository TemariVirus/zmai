//! WIP. Might not end up being used.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

pub fn Tensor(comptime T: type) type {
    return struct {
        allocator: Allocator,
        data: []T,
        shape: []const usize,

        const Self = @This();

        pub fn initRandom(allocator: Allocator, shape: []const usize, comptime random_fn: fn () T) !Self {
            const data = try allocator.alloc(T, len(shape));
            for (data) |*d| {
                d.* = random_fn();
            }

            const shape_copy = try allocator.alloc(usize, shape.len);
            @memcpy(shape_copy, shape);

            return Self{
                .allocator = allocator,
                .data = data,
                .shape = shape_copy,
            };
        }

        pub fn deinit(self: Self) void {
            self.allocator.free(self.data);
            self.allocator.free(self.shape);
        }

        pub fn reshape(self: Self, shape: []const usize) Self {
            assert(len(shape) == len(self.shape));

            return Self{
                .allocator = self.allocator,
                .data = self.data,
                .shape = shape,
            };
        }

        fn len(shape: []const usize) usize {
            var result: usize = 1;
            for (shape) |s| {
                result *= s;
            }
            return result;
        }

        pub fn flatIndex(self: Self, indices: []const usize) usize {
            assert(indices.len == self.shape.len);

            var index: usize = 0;
            var stride: usize = 1;
            for (0..indices.len) |i| {
                assert(indices[i] < self.shape[i]);

                index += indices[i] * stride;
                stride *= self.shape[i];
            }
            return index;
        }

        pub fn get(self: Self, indices: []const usize) T {
            return self.data[self.flatIndex(indices)];
        }

        pub fn set(self: Self, indices: []const usize, value: T) void {
            self.data[self.flatIndex(indices)] = value;
        }

        pub fn apply(self: Self, args: anytype, comptime apply_fn: fn (T, @TypeOf(args)) T) void {
            for (self.data) |*d| {
                d.* = apply_fn(d.*, args);
            }
        }

        // ~120ms on CPU for (500, 500) x (500, 500)
        pub fn matMul(self: Self, other: Self, allocator: Allocator) !Self {
            // TODO: Extend to multiple dimensions
            assert(self.shape.len == 2);
            assert(other.shape.len == 2);
            assert(self.shape[0] == other.shape[1]);

            const result_shape = try allocator.alloc(usize, 2);
            result_shape[0] = self.shape[1];
            result_shape[1] = other.shape[0];

            const result_data = try allocator.alloc(T, Self.len(result_shape));
            const result = Self{
                .allocator = allocator,
                .data = result_data,
                .shape = result_shape,
            };

            for (0..result_shape[0]) |i| {
                for (0..result_shape[1]) |j| {
                    var sum: T = 0;
                    for (0..self.shape[0]) |k| {
                        sum += self.get(&[_]usize{ k, i }) * other.get(&[_]usize{ j, k });
                    }
                    result.set(&[_]usize{ i, j }, sum);
                }
            }

            return result;
        }
    };
}
