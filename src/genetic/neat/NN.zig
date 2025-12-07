//! A neural network as expressed by the *enabled* genes in a genome. Meant for
//! inference only. For a trainable network, use `neat.Genome` instead.

const std = @import("std");
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const fs = std.fs;
const json = std.json;

const neat = @import("../neat.zig");
const ActivationFn = neat.ActivationFn;
const ActivationType = neat.ActivationType;
const Genome = neat.Genome;
const GenomeJson = Genome.GenomeJson;
const Gene = Genome.Gene;

pub const InitOptions = struct {
    input_count: u32,
    output_count: u32,
    hidden_activation: ActivationType = .relu,
    output_activation: ActivationType = .tanh,
};

pub const NNJson = struct {
    options: InitOptions,
    genome: GenomeJson,
};

pub const Node = struct {
    value: f32 = 0,

    pub fn updateValue(
        self: *Node,
        nodes: []Node,
        inputs: []Connection,
        activation: *const ActivationFn,
    ) void {
        self.value = 0;
        for (inputs) |c| {
            self.value += nodes[c.input].value * c.weight;
        }
        self.value = activation(self.value);
    }
};

pub const Connection = struct {
    input: u32,
    weight: f32,
};

/// A jagged array with fixed sizes. Backed by a single contiguous array.
pub fn JaggedArray(comptime T: type) type {
    return struct {
        items: [*]T,
        splits: []u32,

        pub fn init(allocator: Allocator, items: []std.ArrayList(T)) !JaggedArray(T) {
            const splits = try allocator.alloc(u32, items.len + 1);
            errdefer allocator.free(splits);
            splits[0] = 0;
            for (items, 0..) |list, i| {
                splits[i + 1] = @intCast(splits[i] + list.items.len);
            }

            const flat_items = try allocator.alloc(T, splits[splits.len - 1]);
            errdefer allocator.free(flat_items);
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

        pub fn len(self: JaggedArray(T)) usize {
            return self.splits.len - 1;
        }

        pub fn flatLen(self: JaggedArray(T)) usize {
            return self.splits[self.splits.len - 1];
        }

        pub fn deinit(self: JaggedArray(T), allocator: Allocator) void {
            allocator.free(self.items[0..self.flatLen()]);
            allocator.free(self.splits);
        }
    };
}

/// Layout: [...inputs, bias, ...outputs, ...hiddens]
nodes: []Node,
connections: JaggedArray(Connection),
hidden_activation: *const ActivationFn,
output_activation: *const ActivationFn,

const Self = @This();

pub fn init(
    allocator: Allocator,
    genome: Genome,
    options: InitOptions,
    inputs_used: []bool,
) !Self {
    const obj: GenomeJson = .initNoCopy(genome);
    return try fromJson(
        allocator,
        .{ .options = options, .genome = obj },
        inputs_used,
    );
}

/// Loads a neural network directly from a JSON file.
/// `inputs_used` is an output parameter that will be set to a mask indicating
/// which input nodes are used in the network.
pub fn load(allocator: Allocator, path: []const u8, inputs_used: []bool) !Self {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();

    var file_buf: [4096]u8 = undefined;
    var file_reader = file.reader(&file_buf);

    var reader: json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    const saved = try json.parseFromTokenSource(
        NNJson,
        allocator,
        &reader,
        .{ .ignore_unknown_fields = true },
    );
    defer saved.deinit();

    return try fromJson(allocator, saved.value, inputs_used);
}

// TODO: seperate inputs_used from init?
/// Creates a neural network based from a JSON object. `inputs_used` is an
/// output parameter that will be set to a mask indicating which input nodes
/// are used in the network.
pub fn fromJson(allocator: Allocator, obj: NNJson, inputs_used: []bool) !Self {
    const genome = obj.genome.genes;
    const options = obj.options;

    const node_count = blk: {
        var max: u32 = 0;
        for (genome) |gene| {
            max = @max(max, gene.output);
        }
        break :blk max + 1;
    };
    const hidden_offset = options.input_count + options.output_count + 1;

    // Only nodes that can be non-zero are useful
    const useful = try scanForwards(
        allocator,
        options.input_count,
        genome,
        node_count,
    );
    defer allocator.free(useful);
    const used = try scanBackwards(
        allocator,
        options.input_count,
        options.output_count,
        genome,
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
    const node_map = try allocator.alloc(u32, node_count);
    defer allocator.free(node_map);
    var index: u32 = 0;
    for (0..node_count) |i| {
        if (useful[i]) {
            node_map[i] = index;
            index += 1;
        }
    }

    var connection_lists = try allocator.alloc(std.ArrayList(Connection), useful_count);
    defer allocator.free(connection_lists);
    @memset(connection_lists, std.ArrayList(Connection){});
    defer for (connection_lists) |*list| {
        list.deinit(allocator);
    };

    for (genome) |gene| {
        // This implementation is only meant for inference, so we can discard
        // disabled connections
        if (!gene.enabled or !useful[gene.input] or !useful[gene.output]) {
            continue;
        }
        const in = node_map[gene.input];
        const out = node_map[gene.output];
        try connection_lists[out].append(
            allocator,
            .{ .input = in, .weight = gene.weight },
        );
    }
    const connections_arrs: JaggedArray(Connection) = try .init(allocator, connection_lists);
    errdefer connections_arrs.deinit(allocator);

    const nodes = try allocator.alloc(Node, useful_count);
    errdefer allocator.free(nodes);
    @memset(nodes, Node{});

    @memcpy(inputs_used, used[0..inputs_used.len]);
    return Self{
        .nodes = nodes,
        .connections = connections_arrs,
        .hidden_activation = options.hidden_activation.func(),
        .output_activation = options.output_activation.func(),
    };
}

/// The allcator passed in must be the same allocator used to allocate the NN.
pub fn deinit(self: Self, allocator: Allocator) void {
    allocator.free(self.nodes);
    self.connections.deinit(allocator);
}

/// Returns a mask indicating which nodes are affected the inputs.
pub fn scanForwards(
    allocator: Allocator,
    input_count: usize,
    connections: []const Gene,
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

fn scanDownstream(visited: []bool, connections: []const Gene, i: u32) void {
    visited[i] = true;
    for (connections) |c| {
        if (c.input != i) {
            continue;
        }
        if (!visited[c.output] and c.enabled) {
            scanDownstream(visited, connections, c.output);
        }
    }
}

/// Returns a mask indicating which nodes affect the outputs.
pub fn scanBackwards(
    allocator: Allocator,
    input_count: usize,
    output_count: usize,
    connections: []const Gene,
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

fn scanUpstream(visited: []bool, connections: []const Gene, i: u32) void {
    visited[i] = true;
    for (connections) |c| {
        if (c.output != i) {
            continue;
        }
        if (!visited[c.input] and c.enabled) {
            scanUpstream(visited, connections, c.input);
        }
    }
}

/// Feeds the input values through the network and stores the output in the
/// provided slice.
pub fn predict(self: Self, input: []const f32, output: []f32) void {
    const output_offset = input.len + 1;

    // Set input nodes
    for (0..input.len) |i| {
        self.nodes[i].value = input[i];
    }
    self.nodes[input.len].value = 1.0; // Bias node

    // Update hidden all nodes
    for (output_offset + output.len..self.nodes.len) |i| {
        self.nodes[i].updateValue(
            self.nodes,
            self.connections.get(i),
            self.hidden_activation,
        );
    }

    // Update ouput nodes and get output
    for (0..output.len) |i| {
        self.nodes[output_offset + i].updateValue(
            self.nodes,
            self.connections.get(output_offset + i),
            self.output_activation,
        );
        output[i] = self.nodes[output_offset + i].value;
    }
}

test "predict" {
    const allocator = std.testing.allocator;

    // Has used hidden node
    var inputs_used: [8]bool = undefined;
    const nn1 = try load(allocator, "src/genetic/neat/NNs/Qoshae.json", &inputs_used);
    defer nn1.deinit(allocator);

    var out: [2]f32 = undefined;
    nn1.predict(&[8]f32{ 5.2, 1.0, 3.0, 9.0, 11.0, 5.0, 2.0, -0.97 }, &out);
    try expect(out[0] == 0.9761649966239929);
    try expect(out[1] == 0.9984789490699768);

    nn1.predict(&[8]f32{ 2.2, 0.0, 3.0, 5.0, 10.0, 8.0, 4.0, -0.97 }, &out);
    try expect(out[0] == 0.9988278150558472);
    try expect(out[1] == 0.9965899586677551);

    // Has unused hidden node
    const nn2 = try load(allocator, "src/genetic/neat/NNs/Xesa.json", &inputs_used);
    defer nn2.deinit(allocator);

    nn2.predict(&[8]f32{ 5.2, 1.0, 3.0, 9.0, 11.0, 5.0, 2.0, -0.97 }, &out);
    try expect(out[0] == 0.455297589302063);
    try expect(out[1] == -0.9720132350921631);

    nn2.predict(&[8]f32{ 2.2, 0.0, 3.0, 5.0, 10.0, 8.0, 4.0, -0.97 }, &out);
    try expect(out[0] == 1.2168807983398438);
    try expect(out[1] == -0.9620361924171448);
}
