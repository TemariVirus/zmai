//! A simplified neural network implementation for inference only.

const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const json = std.json;

const neat = @import("../neat.zig");
const ActivationFn = neat.ActivationFn;
const ActivationType = neat.ActivationType;
const ConnectionJson = neat.ConnectionJson;
const NNJson = neat.NNJson;

pub const Node = struct {
    value: f32 = 0,
    activation: *const ActivationFn,

    pub fn updateValue(self: *Node, nodes: []Node, inputs: []Connection) void {
        self.value = 0;
        for (inputs) |c| {
            self.value += nodes[c.input].value * c.weight;
        }
        self.value = self.activation(self.value);
    }
};

pub const Connection = struct {
    input: u32,
    weight: f32,
};

pub fn JaggedArray(comptime T: type) type {
    return struct {
        items: [*]T,
        splits: []u32,

        pub fn init(allocator: Allocator, items: []std.ArrayList(T)) !JaggedArray(T) {
            const splits = try allocator.alloc(u32, items.len + 1);
            splits[0] = 0;
            for (items, 0..) |list, i| {
                splits[i + 1] = @intCast(splits[i] + list.items.len);
            }

            const flat_items = try allocator.alloc(T, splits[splits.len - 1]);
            var i: usize = 0;
            for (items) |list| {
                @memcpy(flat_items[i..][0..list.items.len], list.items);
                i += list.items.len;
            }

            return .{
                .items = flat_items.ptr,
                .splits = splits,
            };
        }

        pub fn get(self: JaggedArray(T), index: usize) []T {
            return self.items[self.splits[index]..self.splits[index + 1]];
        }

        pub fn deinit(self: JaggedArray(T), allocator: Allocator) void {
            allocator.free(self.items[0..self.splits[self.splits.len - 1]]);
            allocator.free(self.splits);
        }
    };
}

// Layout: [...inputs, bias, ...outputs, ...hiddens]
nodes: []Node,
connections: JaggedArray(Connection),

const Self = @This();

/// Loads a neural network from a json file.
pub fn load(allocator: Allocator, path: []const u8, inputs_used: []bool) !Self {
    const file = try std.fs.cwd().openFile(path, .{});
    var reader = json.Reader(4096, std.fs.File.Reader).init(allocator, file.reader());
    defer reader.deinit();
    const saved = try json.parseFromTokenSource(NNJson, allocator, &reader, .{
        .ignore_unknown_fields = true,
    });
    defer saved.deinit();

    return try init(
        allocator,
        saved.value.Inputs,
        saved.value.Outputs,
        saved.value.Connections,
        saved.value.Activations,
        inputs_used,
    );
}

fn init(
    allocator: Allocator,
    input_count: usize,
    output_count: usize,
    connections: []ConnectionJson,
    activations: []ActivationType,
    inputs_used: []bool,
) !Self {
    const node_count = blk: {
        var max: u32 = 0;
        for (connections) |c| {
            max = @max(max, c.Output);
        }
        break :blk max + 1;
    };
    const hidden_offset = input_count + output_count + 1;

    // Only nodes that can be non-zero are useful
    const useful = try scanForwards(
        allocator,
        input_count,
        connections,
        node_count,
    );
    defer allocator.free(useful);
    const used = try scanBackwards(
        allocator,
        input_count,
        output_count,
        connections,
        node_count,
    );
    defer allocator.free(used);

    // All inputs and outputs must be kept
    for (0..hidden_offset) |i| {
        useful[i] = true;
    }
    for (hidden_offset..node_count) |i| {
        useful[i] = useful[i] and used[i];
    }

    // Remove non-useful nodes and re-map indices
    const useful_count = blk: {
        var count: usize = 0;
        for (useful) |u| {
            if (u) {
                count += 1;
            }
        }
        break :blk count;
    };
    const nodes = try allocator.alloc(Node, useful_count);
    const node_map = try allocator.alloc(u32, node_count);
    defer allocator.free(node_map);
    var index: u32 = 0;
    for (0..node_count) |i| {
        if (useful[i]) {
            nodes[index] = Node{ .activation = activations[i].func() };
            node_map[i] = index;
            index += 1;
        }
    }

    var connection_lists = try allocator.alloc(std.ArrayList(Connection), nodes.len);
    for (connection_lists) |*list| {
        list.* = std.ArrayList(Connection).init(allocator);
    }
    defer {
        for (connection_lists) |list| {
            list.deinit();
        }
        allocator.free(connection_lists);
    }

    for (connections) |c| {
        // This implementation is only meant for inference, so we can discard
        // disabled connections
        if (!c.Enabled or !useful[c.Input] or !useful[c.Output]) {
            continue;
        }
        try connection_lists[node_map[c.Output]].append(
            .{ .input = c.Input, .weight = c.Weight },
        );
    }
    const connections_arrs = try JaggedArray(Connection).init(allocator, connection_lists);

    @memcpy(inputs_used, used[0..inputs_used.len]);
    return Self{
        .nodes = nodes,
        .connections = connections_arrs,
    };
}

/// The allcator passed in must be the same allocator used to allocate the NN.
pub fn deinit(self: Self, allocator: Allocator) void {
    allocator.free(self.nodes);
    self.connections.deinit(allocator);
}

/// Returns a mask indicating which nodes are affected the inputs.
fn scanForwards(
    allocator: Allocator,
    input_count: usize,
    connections: []ConnectionJson,
    node_count: u32,
) ![]bool {
    const visited = try allocator.alloc(bool, node_count);
    // Visit input nodes
    for (0..input_count + 1) |i| {
        if (!visited[i]) {
            scanDownstream(visited, connections, @intCast(i));
        }
    }
    return visited;
}

fn scanDownstream(visited: []bool, connections: []ConnectionJson, i: u32) void {
    visited[i] = true;
    for (connections) |c| {
        if (c.Input != i) {
            continue;
        }
        if (!visited[c.Output] and c.Enabled) {
            scanDownstream(visited, connections, c.Output);
        }
    }
}

/// Returns a mask indicating which nodes affect the outputs.
fn scanBackwards(
    allocator: Allocator,
    input_count: usize,
    output_count: usize,
    connections: []ConnectionJson,
    node_count: u32,
) ![]bool {
    const visited = try allocator.alloc(bool, node_count);
    // Visit outputs nodes
    for (input_count + 1..input_count + output_count + 1) |i| {
        if (!visited[i]) {
            scanUpstream(visited, connections, @intCast(i));
        }
    }
    return visited;
}

fn scanUpstream(visited: []bool, connections: []ConnectionJson, i: u32) void {
    visited[i] = true;
    for (connections) |c| {
        if (c.Output != i) {
            continue;
        }
        if (!visited[c.Input] and c.Enabled) {
            scanUpstream(visited, connections, c.Input);
        }
    }
}

pub fn predict(self: Self, input: []const f32, output: []f32) void {
    const output_offset = input.len + 1;

    // Set input nodes
    for (0..input.len) |i| {
        self.nodes[i].value = input[i];
    }
    self.nodes[input.len].value = 1.0; // Bias node

    // Update hidden all nodes
    for (input.len + output.len + 1..self.nodes.len) |i| {
        self.nodes[i].updateValue(self.nodes, self.connections.get(i));
    }

    // Update ouput nodes and get output
    for (0..output.len) |i| {
        self.nodes[output_offset + i].updateValue(
            self.nodes,
            self.connections.get(output_offset + i),
        );
        output[i] = self.nodes[output_offset + i].value;
    }
}

test "predict" {
    const allocator = std.testing.allocator;

    // Has used hidden node
    const nn1 = try load(allocator, "src/genetic/neat/NNs/Qoshae.json");
    defer nn1.deinit(allocator);

    var out = nn1.predict([_]f32{ 5.2, 1.0, 3.0, 9.0, 11.0, 5.0, 2.0, -0.97 });
    try expect(out[0] == 0.9761649966239929);
    try expect(out[1] == 0.9984789490699768);

    out = nn1.predict([_]f32{ 2.2, 0.0, 3.0, 5.0, 10.0, 8.0, 4.0, -0.97 });
    try expect(out[0] == 0.9988278150558472);
    try expect(out[1] == 0.9965899586677551);

    // Has unused hidden node
    const nn2 = try load(allocator, "src/genetic/neat/NNs/Xesa.json");
    defer nn2.deinit(allocator);

    out = nn2.predict([_]f32{ 5.2, 1.0, 3.0, 9.0, 11.0, 5.0, 2.0, -0.97 });
    try expect(out[0] == 0.455297589302063);
    try expect(out[1] == -0.9720132350921631);

    out = nn2.predict([_]f32{ 2.2, 0.0, 3.0, 5.0, 10.0, 8.0, 4.0, -0.97 });
    try expect(out[0] == 1.2168807983398438);
    try expect(out[1] == -0.9620361924171448);
}
