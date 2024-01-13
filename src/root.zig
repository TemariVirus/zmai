const std = @import("std");
pub const supervised = @import("supervised.zig");

var rand = std.rand.DefaultPrng.init(0);
const random = rand.random();

pub fn setRandomSeed(seed: u64) void {
    rand.seed(seed);
}

pub fn gaussianRandom() f32 {
    return random.floatNorm(f32);
}
