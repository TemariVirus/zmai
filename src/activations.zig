pub const Activation = enum {
    identity,
    sigmoid,
    relu,
    leaky_relu,
    elu,
    selu,
};

/// Calculates the values in `arr` after applying the activation function, and
/// stores the result in `arr`.
pub fn forward(activation: Activation, arr: []f32) void {
    switch (activation) {
        .identity => identity(arr),
        .sigmoid => sigmoid(arr),
        .relu => relu(arr),
        .leaky_relu => leakyRelu(arr),
        .elu => elu(arr),
        .selu => selu(arr),
    }
}

/// Calculates the partial derivatives of the values in `arr`, and stores the
/// result in `arr`, assuming that the values in `arr` have already been passed
/// through the activation function.
pub fn backward(activation: Activation, arr: []f32) void {
    switch (activation) {
        .identity => identityBackward(arr),
        .sigmoid => sigmoidBackward(arr),
        .relu => reluBackward(arr),
        .leaky_relu => leakyReluBackward(arr),
        .elu => eluBackward(arr),
        .selu => seluBackward(arr),
    }
}

pub fn identity(arr: []f32) void {
    _ = arr;
}

pub fn identityBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* = 1;
    }
}

pub fn sigmoid(arr: []f32) void {
    for (arr) |*x| {
        x.* = 1 / (1 + @exp(-x.*));
    }
}

pub fn sigmoidBackward(arr: []f32) void {
    for (arr) |*x| {
        x.* *= 1 - x.*;
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

pub const LEAKY_RELU_A = 0.1;
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
