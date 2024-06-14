//! NeuralEvolution of Augmenting Topologies.

pub const NN = @import("neat/NN.zig");
pub const Genome = @import("neat/Genome.zig");
pub const Trainer = @import("neat/Trainer.zig");

const std = @import("std");
const math = std.math;

pub const ActivationFn = fn (f32) f32;
pub const ActivationType = enum(u8) {
    sigmoid = 0,
    tanh = 1,
    relu = 2,
    leaky_relu = 3,
    isru = 4,
    elu = 5,
    selu = 6,
    gelu = 7,
    softplus = 8,
    identity = 9,
    swish = 10,
    step = 11,
    gaussian = 12,

    pub fn func(self: ActivationType) *const ActivationFn {
        return switch (self) {
            .sigmoid => Activation.sigmoid,
            .tanh => Activation.tanh,
            .relu => Activation.relu,
            .leaky_relu => Activation.leakyRelu,
            .isru => Activation.isru,
            .elu => Activation.elu,
            .selu => Activation.selu,
            .gelu => Activation.gelu,
            .softplus => Activation.softplus,
            .identity => Activation.identity,
            .swish => Activation.swish,
            .step => Activation.step,
            .gaussian => Activation.gaussian,
        };
    }
};

pub const Activation = struct {
    pub fn sigmoid(x: f32) f32 {
        return 1.0 / (1.0 + @exp(-x));
    }

    pub fn tanh(x: f32) f32 {
        return math.tanh(x);
    }

    pub fn relu(x: f32) f32 {
        return if (x >= 0) x else 0;
    }

    pub fn leakyRelu(x: f32) f32 {
        return if (x >= 0) x else 0.01 * x;
    }

    pub fn isru(x: f32) f32 {
        return if (x >= 0) x else x / @sqrt(x * x + 1);
    }

    pub fn elu(x: f32) f32 {
        return if (x >= 0) x else @exp(x) - 1;
    }

    pub fn selu(x: f32) f32 {
        return 1.05070098735548 * if (x >= 0) x else 1.670086994173469 * (@exp(x) - 1);
    }

    pub fn gelu(x: f32) f32 {
        return x / (1.0 + @exp(-1.702 * x));
    }

    pub fn softplus(x: f32) f32 {
        return @log(1 + @exp(x));
    }

    pub fn identity(x: f32) f32 {
        return x;
    }

    pub fn swish(x: f32) f32 {
        return x / (1 + @exp(-x));
    }

    pub fn step(x: f32) f32 {
        return @floatFromInt(@intFromBool(x >= 0));
    }

    pub fn gaussian(x: f32) f32 {
        return @exp(-x * x);
    }
};

test {
    @import("std").testing.refAllDecls(@This());
}
