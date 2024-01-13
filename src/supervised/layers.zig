const std = @import("std");
const assert = std.debug.assert;

pub const Dense = @import("layers/Dense.zig");

pub const LayerTag = enum {
    dense,

    // TODO
    // conv2d,
    // max_pool2d,
    // avg_pool2d,
};

pub const Layer = union(LayerTag) {
    dense: Dense,

    /// The number of input neurons.
    pub fn inputSize(self: Layer) usize {
        return switch (self) {
            .dense => |layer| layer.inputSize(),
        };
    }

    /// The number of output neurons.
    pub fn outputSize(self: Layer) usize {
        return switch (self) {
            .dense => |layer| layer.outputSize(),
        };
    }

    /// The number of trainable parameters.
    pub fn size(self: Layer) usize {
        return switch (self) {
            .dense => |layer| layer.size(),
        };
    }

    /// Does a forward pass and stores the activations in `output`.
    pub fn forward(self: Layer, input: []const f32, output: []f32) void {
        switch (self) {
            .dense => |layer| layer.forward(input, output),
        }
    }

    /// Does a backwards pass and updates the weights and biases of this layer.
    ///
    /// `learning_rate` - the learning rate to use.
    ///
    /// `input` - the input to this layer in the forward pass. This function
    /// updates this slice to be the gradient of the loss function with respect to
    /// the input of this layer.
    ///
    /// `output` - the activations of this layer in the forward pass. This function
    /// updates this slice to be the gradient of the activations with respect to
    /// the unactivated output of this layer.
    ///
    /// `output_grad` - the gradient of the loss function with respect to the
    /// activations of this layer.
    ///
    /// `deltas` - the deltas to add to the weights and biases of this layer. This
    /// function updates this slice accordingly, but does not apply the deltas.
    pub fn backward(
        self: Layer,
        input: []f32,
        output: []f32,
        output_grad: []const f32,
        deltas: []f32,
    ) void {
        switch (self) {
            .dense => |layer| layer.backward(
                input,
                output,
                output_grad,
                deltas,
            ),
        }
    }

    /// Updates the weights and biases of this layer.
    pub fn update(self: Layer, deltas: []const f32, learning_rate: f32) void {
        switch (self) {
            .dense => |layer| layer.update(deltas, learning_rate),
        }
    }
};
