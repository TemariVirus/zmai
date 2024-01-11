const std = @import("std");
const assert = std.debug.assert;

pub const Dense = @import("layers/Dense.zig");

pub const Layer = struct {
    ptr: *anyopaque,
    forwardFn: *const fn (ptr: *anyopaque, input: []const f32, output: []f32) void,
    input_size: usize,
    output_size: usize,

    const Self = @This();

    pub fn init(
        pointer: anytype,
        comptime forwardFn: fn (@TypeOf(pointer), []const f32, []f32) void,
        input_size: usize,
        output_size: usize,
    ) Self {
        const Ptr = @TypeOf(pointer);
        assert(@typeInfo(Ptr) == .Pointer);
        assert(@typeInfo(Ptr).Pointer.size == .One);
        assert(@typeInfo(@typeInfo(Ptr).Pointer.child) == .Struct);
        const gen = struct {
            fn forward(ptr: *anyopaque, input: []const f32, output: []f32) void {
                const self: Ptr = @ptrCast(@alignCast(ptr));
                forwardFn(self, input, output);
            }
        };

        return .{
            .ptr = pointer,
            .forwardFn = gen.forward,
            .input_size = input_size,
            .output_size = output_size,
        };
    }

    pub fn forward(self: Self, input: []const f32, output: []f32) void {
        self.forwardFn(self.ptr, input, output);
    }
};
