//! NeuralEvolution of Augmenting Topologies.

pub const FastNN = @import("neat/FastNN.zig");

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
};

pub const ConnectionJson = struct {
    Enabled: bool,
    Input: u32,
    Output: u32,
    Weight: f32,
};

pub const NNJson = struct {
    Name: []const u8,
    Played: bool,
    Inputs: usize,
    Outputs: usize,
    Fitness: f64,
    Connections: []ConnectionJson,
    Activations: []ActivationType,
};

test {
    @import("std").testing.refAllDecls(@This());
}
