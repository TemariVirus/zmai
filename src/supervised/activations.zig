const math = @import("std").math;

pub const Activation = enum {
    identity,
    sigmoid,
    tanh,
    softmax,
    relu,
    leaky_relu,
    elu,
    selu,

    /// Calculates the values in `arr` after applying the activation function, and
    /// stores the result in `arr`.
    pub fn forward(self: Activation, arr: []f32) void {
        switch (self) {
            .identity => identity(arr),
            .sigmoid => sigmoid(arr),
            .tanh => tanh(arr),
            .softmax => softmax(arr),
            .relu => relu(arr),
            .leaky_relu => leakyRelu(arr),
            .elu => elu(arr),
            .selu => selu(arr),
        }
    }

    /// Calculates the partial derivatives of the values in `arr`, and stores the
    /// result in `arr`, assuming that the values in `arr` have already been passed
    /// through the activation function.
    pub fn backward(self: Activation, arr: []f32) void {
        switch (self) {
            .identity => identityBackward(arr),
            .sigmoid => sigmoidBackward(arr),
            .tanh => tanhBackward(arr),
            .softmax => softmaxBackward(arr),
            .relu => reluBackward(arr),
            .leaky_relu => leakyReluBackward(arr),
            .elu => eluBackward(arr),
            .selu => seluBackward(arr),
        }
    }
};

/// Runs in O(0) time. That's even better than O(1)!
pub fn identity(arr: []f32) void {
    _ = arr;
}

pub fn identityBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* = 1;
    }
}

pub const SIGMOID_A = 4.0;
pub fn sigmoid(arr: []f32) void {
    for (arr) |*x| {
        x.* = 1 / (1 + @exp(-SIGMOID_A * x.*));
    }
}

pub fn sigmoidBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* *= SIGMOID_A * (1 - x.*);
    }
}

pub fn tanh(arr: []f32) void {
    for (arr) |*x| {
        x.* = math.tanh(x.*);
    }
}

pub fn tanhBackward(arr: []f32) void {
    for (arr) |*x| {
        const tanh2 = x.* * x.*;
        x.* = 1 - tanh2;
    }
}

pub fn softmax(arr: []f32) void {
    // Softmax is invariant under translation, so make the largest value 0 to
    // avoid infinities
    var max = arr[0];
    for (arr[1..]) |x| {
        max = @max(max, x);
    }

    var sum: f32 = 0.0;
    for (arr) |*x| {
        x.* = @exp(x.* - max);
        sum += x.*;
    }
    for (arr) |*x| {
        x.* /= sum;
    }
}

/// NOTE: Assumes that softmax is only used in the last layer and that the loss
/// function is cross-entropy. Thus, this function simply fills the array with
/// ones as the combined derivative of softmax and cross-entropy is calculated
/// in `losses.crossEntropyBackward`.
pub fn softmaxBackward(arr: []f32) void {
    for (0..arr.len) |i| {
        // arr[i] *= 1 - arr[i]; // Actual
        arr[i] = 1.0;
    }
}

pub fn relu(arr: []f32) void {
    for (arr) |*x| {
        x.* = @max(x.*, 0);
    }
}

pub fn reluBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0) 1 else 0;
    }
}

pub const LEAKY_RELU_A = 0.3;
pub fn leakyRelu(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0)
            x.*
        else
            LEAKY_RELU_A * x.*;
    }
}

pub fn leakyReluBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0)
            1
        else
            LEAKY_RELU_A;
    }
}

pub const ELU_A = 1.0;
pub fn elu(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0)
            x.*
        else
            ELU_A * (@exp(x.*) - 1);
    }
}

pub fn eluBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0)
            1
        else
            x.* + ELU_A;
    }
}

pub const SELU_A = 1.6732632423543772848170429916717;
pub const SELU_LAMBDA = 1.0507009873554804934193349852946;
pub fn selu(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0)
            SELU_LAMBDA * x.*
        else
            SELU_LAMBDA * SELU_A * (@exp(x.*) - 1);
    }
}

pub fn seluBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* = if (x.* > 0)
            SELU_LAMBDA
        else
            SELU_LAMBDA * (x.* + SELU_A);
    }
}
