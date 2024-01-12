const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

pub const Loss = enum {
    mean_absolute_error,
    mean_squared_error,
    cross_entropy,
};

pub fn forward(loss: Loss, y_pred: []const f32, y_true: []const f32) f32 {
    return switch (loss) {
        .mean_absolute_error => mae(y_pred, y_true),
        .mean_squared_error => mse(y_pred, y_true),
        .cross_entropy => crossEntropy(y_pred, y_true),
    };
}

pub fn backward(loss: Loss, y_pred: []const f32, y_true: []f32) void {
    return switch (loss) {
        .mean_absolute_error => maeBackward(y_pred, y_true),
        .mean_squared_error => mseBackward(y_pred, y_true),
        .cross_entropy => crossEntropyBackward(y_pred, y_true),
    };
}

pub fn mae(y_pred: []const f32, y_true: []const f32) f32 {
    assert(y_pred.len == y_true.len);

    var sum: f32 = 0.0;
    for (0..y_true.len) |i| {
        sum += @abs(y_pred[i] - y_true[i]);
    }
    return sum / @as(f32, @floatFromInt(y_true.len));
}

pub fn maeBackward(y_pred: []f32, y_true: []f32) void {
    assert(y_pred.len == y_true.len);

    for (0..y_true.len) |i| {
        y_pred[i] = math.sign(y_pred[i] - y_true[i]);
    }
}

pub fn mse(y_pred: []const f32, y_true: []const f32) f32 {
    assert(y_pred.len == y_true.len);

    var sum: f32 = 0.0;
    for (0..y_true.len) |i| {
        sum += (y_pred[i] - y_true[i]) * (y_pred[i] - y_true[i]);
    }
    return sum / @as(f32, @floatFromInt(y_true.len));
}

pub fn mseBackward(y_pred: []f32, y_true: []f32) void {
    assert(y_pred.len == y_true.len);

    for (0..y_true.len) |i| {
        y_pred[i] = 2.0 * (y_pred[i] - y_true[i]);
    }
}

pub fn crossEntropy(y_pred: []const f32, y_true: []const f32) f32 {
    assert(y_pred.len == y_true.len);

    var sum: f32 = 0.0;
    for (0..y_true.len) |i| {
        sum -= y_true[i] * @log2(y_pred[i]);
    }
    return sum;
}

pub fn crossEntropyBackward(y_pred: []f32, y_true: []f32) void {
    assert(y_pred.len == y_true.len);

    for (0..y_true.len) |i| {
        y_pred[i] = -y_true[i] / (y_pred[i] * @log(2.0));
    }
}
