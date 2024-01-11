pub const Activation = *const fn ([]f32) void;

pub fn sigmoid(arr: []f32) void {
    for (arr) |*x| {
        x.* = 1 / (1 + @exp(-x.*));
    }
}

pub fn relu(arr: []f32) void {
    for (arr) |*x| {
        x.* = @max(x.*, 0);
    }
}

pub fn leakyRelu(arr: []f32) void {
    const a = 0.1;

    for (arr) |*x| {
        x.* = if (x.* > 0)
            x.*
        else
            a * x.*;
    }
}

pub fn elu(arr: []f32) void {
    const a = 1.0;

    for (arr) |*x| {
        x.* = if (x.* > 0)
            x.*
        else
            a * (@exp(x.*) - 1);
    }
}

pub fn selu(arr: []f32) void {
    const a = 1.6732632423543772848170429916717;
    const b = 1.0507009873554804934193349852946;

    for (arr) |*x| {
        x.* = if (x.* > 0)
            x.*
        else
            a * (@exp(x.*) - 1);
        x.* *= b;
    }
}
