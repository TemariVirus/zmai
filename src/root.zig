pub const supervised = @import("supervised.zig");

const std = @import("std");

var rand = std.rand.DefaultPrng.init(0);
pub const random = rand.random();

/// Sets the seed used for random number generation.
pub fn setRandomSeed(seed: u64) void {
    rand.seed(seed);
}

/// Returns a number drawn from a uniform distribution in the range [-1, 1).
pub fn uniformRandom() f32 {
    return random.float(f32) * 2 - 1;
}

/// Returns a number drawn from a normal distribution with mean 0 and standard
/// deviation 1.
pub fn gaussianRandom() f32 {
    return random.floatNorm(f32);
}

test {
    std.testing.refAllDecls(@This());
}
