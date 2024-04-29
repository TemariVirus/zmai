const std = @import("std");
const Allocator = std.mem.Allocator;
const Client = std.http.Client;
const gzip = std.compress.gzip;
const assert = std.debug.assert;

const zmai = @import("zmai");
const supervised = zmai.supervised;
const Model = supervised.Model;
const Sgd = supervised.optimizers.Sgd;

const Layer = supervised.layers.Layer;
const Conv2D = supervised.layers.Conv2D;
const MaxPool2D = supervised.layers.MaxPool2D;
const AvgPool2D = supervised.layers.AvgPool2D;
const Dense = supervised.layers.Dense;

// Crop edges of images as they are mostly 0s
const EDGE_CROP = 2;
const IMAGE_WIDTH = 28 - (2 * EDGE_CROP);
const IMAGE_HEIGHT = 28 - (2 * EDGE_CROP);
const NUM_CLASSES = 10;

// Train a model to recognize handwritten digits from the MNIST dataset.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    // Define the model
    zmai.setRandomSeed(23);
    const conv1 = try Conv2D.init(
        allocator,
        .{ .x = IMAGE_WIDTH, .y = IMAGE_HEIGHT, .z = 1 },
        3,
        .{ .x = 5, .y = 5 },
        .{ .x = 1, .y = 1 },
        .elu,
        zmai.uniformRandom,
    );
    const pool1 = MaxPool2D.init(
        conv1.outputShape(),
        .{ .x = 2, .y = 2 },
        .{ .x = 2, .y = 2 },
    );
    const conv2 = try Conv2D.init(
        allocator,
        pool1.outputShape(),
        6,
        .{ .x = 3, .y = 3 },
        .{ .x = 1, .y = 1 },
        .elu,
        zmai.uniformRandom,
    );
    const pool2 = AvgPool2D.init(
        conv2.outputShape(),
        .{ .x = 2, .y = 2 },
        .{ .x = 2, .y = 2 },
    );
    const dense1 = try Dense.init(
        allocator,
        pool2.outputSize(),
        24,
        .elu,
        zmai.uniformRandom,
    );
    const dense2 = try Dense.init(
        allocator,
        dense1.outputSize(),
        NUM_CLASSES,
        .softmax,
        zmai.uniformRandom,
    );
    defer conv1.deinit(allocator);
    defer conv2.deinit(allocator);
    defer dense1.deinit(allocator);
    defer dense2.deinit(allocator);

    const layers = [_]Layer{
        .{ .conv2d = conv1 },
        .{ .max_pool2d = pool1 },
        .{ .conv2d = conv2 },
        .{ .avg_pool2d = pool2 },
        .{ .dense = dense1 },
        .{ .dense = dense2 },
    };
    const model = Model{
        .layers = &layers,
    };

    std.debug.print("Loading dataset...\n", .{});
    const x_train, const y_train, const x_test, const y_test = try loadMinst(allocator);
    defer {
        for (x_train, y_train) |x, y| {
            allocator.free(x);
            allocator.free(y);
        }
        for (x_test, y_test) |x, y| {
            allocator.free(x);
            allocator.free(y);
        }
        allocator.free(x_train);
        allocator.free(y_train);
        allocator.free(x_test);
        allocator.free(y_test);
    }

    std.debug.print("Training model...\n", .{});
    const sgd = try Sgd.init(allocator, model);
    defer sgd.deinit();
    try sgd.fit(
        x_train,
        y_train,
        15,
        60,
        .cross_entropy,
        0.2,
    );

    // Print train and test accuracy
    const train_acc = try accuracy(allocator, model, x_train, y_train);
    std.debug.print("Train accuracy: {d:.2}%\n", .{train_acc * 100});

    const test_acc = try accuracy(allocator, model, x_test, y_test);
    std.debug.print("Test accuracy: {d:.2}%\n", .{test_acc * 100});
}

fn loadMinst(allocator: Allocator) !struct {
    [][]f32,
    [][]f32,
    [][]f32,
    [][]f32,
} {
    var client = Client{ .allocator = allocator };
    defer client.deinit();

    const x_train = try readMinstImages(
        allocator,
        &client,
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    );
    errdefer allocator.free(x_train);
    const y_train = try readMinstLabels(
        allocator,
        &client,
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    );
    errdefer allocator.free(y_train);
    const x_test = try readMinstImages(
        allocator,
        &client,
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    );
    errdefer allocator.free(x_test);
    const y_test = try readMinstLabels(
        allocator,
        &client,
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    );

    return .{ x_train, y_train, x_test, y_test };
}

fn readMinstLabels(
    allocator: Allocator,
    client: *Client,
    url: []const u8,
) ![][]f32 {
    var res_body = std.ArrayList(u8).init(allocator);
    defer res_body.deinit();

    const result = try client.fetch(.{
        .response_storage = .{ .dynamic = &res_body },
        .location = .{ .url = url },
    });
    if (result.status != .ok) {
        return error.NetworkError;
    }

    var result_stream = std.io.FixedBufferStream([]u8){
        .buffer = res_body.items,
        .pos = 0,
    };

    var decompresser = gzip.decompressor(result_stream.reader());
    const decompressed_reader = decompresser.reader();

    // Check magic number
    assert(try decompressed_reader.readInt(i32, .big) == 2049);

    const len = try decompressed_reader.readInt(u32, .big);
    const one_hot = try allocator.alloc([]f32, len);
    for (one_hot) |*row| {
        const index = try decompressed_reader.readByte();
        row.* = try allocator.alloc(f32, 10);
        for (0..row.len) |i| {
            row.*[i] = if (i == index) 1.0 else 0.0;
        }
    }

    return one_hot;
}

fn readMinstImages(
    allocator: Allocator,
    client: *Client,
    url: []const u8,
) ![][]f32 {
    var res_body = std.ArrayList(u8).init(allocator);
    defer res_body.deinit();

    const result = try client.fetch(.{
        .response_storage = .{ .dynamic = &res_body },
        .max_append_size = 10 * 1024 * 1024, // Largest file is around 9.5MB, 10MiB should be enough
        .location = .{ .url = url },
    });
    if (result.status != .ok) {
        return error.NetworkError;
    }

    var result_stream = std.io.FixedBufferStream([]const u8){
        .buffer = res_body.items,
        .pos = 0,
    };

    var decompresser = gzip.decompressor(result_stream.reader());
    const decompressed_reader = decompresser.reader();

    // Check magic number
    assert(try decompressed_reader.readInt(i32, .big) == 2051);

    const len = try decompressed_reader.readInt(u32, .big);
    _ = try decompressed_reader.readInt(u32, .big); // rows = 28
    _ = try decompressed_reader.readInt(u32, .big); // cols = 28

    const images = try allocator.alloc([]f32, len);
    for (images) |*row| {
        var buffer = [_]u8{undefined} ** (28 * 28);
        if (try decompressed_reader.readAll(&buffer) != buffer.len) {
            return error.RanOutOfData;
        }

        row.* = try allocator.alloc(f32, IMAGE_WIDTH * IMAGE_HEIGHT);
        for (0..IMAGE_HEIGHT) |i| {
            const y = i + EDGE_CROP;
            for (0..IMAGE_WIDTH) |j| {
                const x = j + EDGE_CROP;
                row.*[i * IMAGE_WIDTH + j] = @as(f32, @floatFromInt(buffer[y * 28 + x])) / 255.0;
            }
        }
    }

    return images;
}

fn accuracy(
    allocator: Allocator,
    model: Model,
    x_data: []const []const f32,
    y_data: []const []const f32,
) !f32 {
    assert(x_data.len == y_data.len);

    var correct: usize = 0;
    for (x_data, y_data) |x, y| {
        const true_label = maxIndex(y);

        const y_pred = try model.predict(allocator, x);
        const pred_label = maxIndex(y_pred);
        allocator.free(y_pred);

        if (true_label == pred_label) {
            correct += 1;
        }
    }

    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(x_data.len));
}

fn maxIndex(arr: []const f32) usize {
    var max = arr[0];
    var max_i: usize = 0;
    for (1..arr.len) |i| {
        if (arr[i] > max) {
            max = arr[i];
            max_i = i;
        }
    }

    return max_i;
}
