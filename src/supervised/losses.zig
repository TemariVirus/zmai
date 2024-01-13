const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

pub const Loss = enum {
    mean_absolute_error,
    mean_squared_error,
    cross_entropy,

    pub fn forward(loss: Loss, y_pred: []const f32, y_true: []const f32) f32 {
        return switch (loss) {
            .mean_absolute_error => mae(y_pred, y_true),
            .mean_squared_error => mse(y_pred, y_true),
            .cross_entropy => crossEntropy(y_pred, y_true),
        };
    }

    pub fn backward(loss: Loss, y_pred: []f32, y_true: []const f32) void {
        return switch (loss) {
            .mean_absolute_error => maeBackward(y_pred, y_true),
            .mean_squared_error => mseBackward(y_pred, y_true),
            .cross_entropy => crossEntropyBackward(y_pred, y_true),
        };
    }
};

pub fn mae(y_pred: []const f32, y_true: []const f32) f32 {
    assert(y_pred.len == y_true.len);

    var sum: f32 = 0.0;
    for (0..y_true.len) |i| {
        sum += @abs(y_pred[i] - y_true[i]);
    }
    return sum / @as(f32, @floatFromInt(y_true.len));
}

pub fn maeBackward(y_pred: []f32, y_true: []const f32) void {
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

pub fn mseBackward(y_pred: []f32, y_true: []const f32) void {
    assert(y_pred.len == y_true.len);

    for (0..y_true.len) |i| {
        y_pred[i] = 2.0 * (y_pred[i] - y_true[i]);
    }
}

pub fn crossEntropy(y_pred: []const f32, y_true: []const f32) f32 {
    assert(y_pred.len == y_true.len);

    var sum: f32 = 0.0;
    for (0..y_true.len) |i| {
        assert(y_pred[i] >= 0.0 and y_pred[i] <= 1.0);
        assert(y_true[i] >= 0.0 and y_true[i] <= 1.0);

        sum -= y_true[i] * @log2(@max(math.floatEps(f32), y_pred[i]));
    }
    return sum;
}

/// NOTE: Assumes that softmax is only used in the last layer and that the loss
/// function is cross-entropy. Thus, this function calculates the combined
/// derivative of softmax and cross-entropy.
//
// Let p be the true probabilities, q be the predicted probabilities, and z be
// the logits.
//
// q_i          = e^z_i / (e^z_1 + e^z_2 + ...)
// loss         = -(p_1 * log2(q_1) + p_2 * log2(q_2) + ...)
//
// ∂q_i / ∂z_i  = q_i(1 - q_i)
// ∂q_i / ∂z_j  = -q_j * q_i            {i != j}
// ∂loss / ∂q_i = -p_i / (q_i * ln(2))
//
// ∂loss / ∂z_i = (∂loss / ∂q_1 * ∂q_1 / ∂z_i
//                + ∂loss / ∂q_2 * ∂q_2 / ∂z_i
//                + ...)
//              = (-p_i / (q_i * ln(2)) * q_i(1 - q_i)
//                + -p_1 / (q_1 * ln(2)) * -q_i * q_1
//                + -p_2 / (q_2 * ln(2)) * -q_i * q_2
//                + ...)
//              = (-p_i(1 - q_i) + q_i(p_1 + p_2 + ...)) / ln(2)
//              = (-p_i(1 - q_i) + q_i(1 - p_i)) / ln(2)
//              = (q_i - p_i) / ln(2)
pub fn crossEntropyBackward(y_pred: []f32, y_true: []const f32) void {
    assert(y_pred.len == y_true.len);

    for (0..y_true.len) |i| {
        assert(y_pred[i] >= 0.0 and y_pred[i] <= 1.0);
        assert(y_true[i] >= 0.0 and y_true[i] <= 1.0);

        // y_pred[i] = -y_true[i] / @max(math.floatMin(f32), y_pred[i] * @log(2.0)); // Actual
        y_pred[i] = (y_pred[i] - y_true[i]) / @log(2.0);
    }
}
