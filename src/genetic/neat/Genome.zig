//! Stores the set of genes that represent a genome. Meant for reproduction
//! during training only. To get a usable neural network for inference, use
//! `neat.NN` instead.

const std = @import("std");
const Allocator = std.mem.Allocator;

const root = @import("../../root.zig");
const random = root.random;
const GeneRecord = @import("../neat.zig").Trainer.GeneRecord;

pub const Gene = struct {
    input: u32,
    output: u32,
    weight: f32,
    enabled: bool,
};
pub const GeneMap = std.AutoArrayHashMapUnmanaged(u32, Gene);

/// A JSON serializable representation of a genome.
pub const GenomeJson = struct {
    genes: []const Gene,

    /// Creates a JSON serializable object from a genome. The object owns its
    /// data and must be deallocated with `deinit`.
    pub fn init(allocator: Allocator, genome: Self) !GenomeJson {
        return .{
            .genes = try allocator.dupe(Gene, genome.genes.values()),
        };
    }

    /// Creates a JSON serializable object from a genome without allocation.
    /// Modifying the genome may invalidate the object. `deinit` should not be
    /// called when initialized this way.
    pub fn initNoCopy(genome: Self) GenomeJson {
        return .{
            .genes = genome.genes.values(),
        };
    }

    pub fn deinit(self: GenomeJson, allocator: Allocator) void {
        allocator.free(self.genes);
    }
};

pub const MateOptions = struct {
    /// The probability of copying a gene's weight from the less fit parent.
    bad_weight_prob: f32 = 0.3,
    /// The probability of copying a gene's enabled state from the less fit
    /// parent.
    bad_enabled_prob: f32 = 0.1,
};

pub const MutateOptions = struct {
    /// Allow recursive connections with cycle length 2 or less.
    recursive_networks: bool = false,
    /// The probability of perturbing a weight.
    weight_purturb_prob: f32 = 0.8,
    /// The maximum magnitude of additive weight purtabations.
    weight_translate_power: f32 = 0.12,
    /// The magnitude of multiplicative weight purtabations.
    weight_scale_power: f32 = 0.075,
    /// The probability of enabling/disabling a connection.
    enable_toggle_prob: f32 = 0.1,
    /// The minimum weight value.
    weight_min: ?f32 = null,
    /// The maximum weight value.
    weight_max: ?f32 = null,
    /// The probability of adding a connection between existing nodes.
    connection_add_prob: f32 = 0.1,
    /// The probability of adding a node between existing connections.
    node_add_prob: f32 = 0.02,
    /// The number of times to try when adding a new node or connection.
    try_add_times: u32 = 20,
};

pub const DistanceOptions = struct {
    /// The coefficient to multiply the weight difference by.
    wight_diff_weight: f32 = 1,
    /// The coefficient to multiply the number of disjoint or ecxess genes by.
    non_matching_weight: f32 = 2,
    /// Genomes with less than this number of connections will not have their
    /// number of excess/disjoint connections normalized.
    small_genome_bias: usize = 20,
};

const Self = @This();

genes: GeneMap,

/// Initializes a minimal network with random weights.
pub fn init(
    allocator: Allocator,
    input_count: u32,
    output_count: u32,
    gene_record: *GeneRecord,
) !Self {
    var genes = GeneMap{};
    errdefer genes.deinit(allocator);

    // Connect every input to every output
    // Add 1 to input_count for bias node
    for (0..input_count + 1) |in| {
        for (0..output_count) |out| {
            const gene = Gene{
                .input = @intCast(in),
                .output = @intCast(input_count + 1 + out),
                .weight = root.gaussianRandom(),
                .enabled = true,
            };
            const id = try gene_record.putGet(gene);
            try genes.putNoClobber(allocator, id, gene);
        }
    }

    return .{
        .genes = genes,
    };
}

/// Deserializes a genome from a JSON object.
pub fn from(allocator: Allocator, obj: GenomeJson, gene_record: *GeneRecord) !Self {
    var genes = GeneMap{};
    errdefer genes.deinit(allocator);

    try genes.entries.resize(allocator, obj.genes.len);
    for (0..obj.genes.len) |i| {
        genes.keys()[i] = try gene_record.putGet(obj.genes[i]);
    }
    @memcpy(genes.values(), obj.genes);

    try genes.reIndex(allocator);
    return .{ .genes = genes };
}

pub fn deinit(self: *Self, allocator: Allocator) void {
    self.genes.deinit(allocator);
}

pub fn mateWith(
    self: Self,
    allocator: Allocator,
    less_fit: Self,
    options: MateOptions,
) !Self {
    var child = try self.clone(allocator);
    var items = child.genes.iterator();
    while (items.next()) |item| {
        const other_con = less_fit.genes.get(item.key_ptr.*) orelse continue;
        if (random.float(f32) < options.bad_weight_prob) {
            item.value_ptr.weight = other_con.weight;
        }
        if (random.float(f32) < options.bad_enabled_prob) {
            item.value_ptr.enabled = other_con.enabled;
        }
    }

    return child;
}

pub fn mutate(
    self: *Self,
    allocator: Allocator,
    gene_record: *GeneRecord,
    options: MutateOptions,
) !void {
    // Mutate connections (genes)
    for (self.genes.keys()) |k| {
        const gene = self.genes.getPtr(k) orelse unreachable;
        // Mutate the weight
        if (random.float(f32) < options.weight_purturb_prob) {
            switch (random.weightedIndex(u32, &.{ 10, 45, 45 })) {
                0 => gene.weight = root.gaussianRandom(),
                1 => gene.weight += root.uniformRandom() * options.weight_translate_power,
                2 => gene.weight *= 1 + root.uniformRandom() * options.weight_scale_power,
                else => unreachable,
            }
        }
        // Enable/disable gene
        if (random.float(f32) < options.enable_toggle_prob) {
            gene.enabled = !gene.enabled;
        }
        // Keep weight within bounds
        if (options.weight_min) |min| {
            gene.weight = @max(gene.weight, min);
        }
        if (options.weight_max) |max| {
            gene.weight = @min(gene.weight, max);
        }
    }

    // Add connection between existing nodes
    if (random.float(f32) < options.connection_add_prob) {
        for (0..options.try_add_times) |_| {
            const in = random.uintLessThan(u32, self.nodeCount());
            const out = random.uintLessThan(u32, self.nodeCount());
            const new_gene = Gene{
                .input = in,
                .output = out,
                .weight = root.gaussianRandom(),
                .enabled = true,
            };
            const id = try gene_record.putGet(new_gene);

            // Check for reverse connection (or itself, in the case in == out)
            if (gene_record.get(.{
                .input = out,
                .output = in,
                .weight = undefined,
                .enabled = undefined,
            })) |reverse_id| {
                if (!options.recursive_networks and
                    (id == reverse_id or self.genes.contains(reverse_id)))
                {
                    continue;
                }
            }

            // Only add gene if it isn't already part of the genome
            const result = try self.genes.getOrPut(allocator, id);
            if (result.found_existing) {
                continue;
            }
            result.value_ptr.* = new_gene;
            break;
        }
    }

    // Add node onto existing connection
    if (random.float(f32) < options.node_add_prob) {
        // Find connection to split
        const split_gene = for (0..options.try_add_times) |_| {
            const index = random.uintLessThan(usize, self.genes.count());
            const split_con = self.genes.getPtr(self.genes.keys()[index]) orelse unreachable;
            if (!split_con.enabled) {
                continue;
            }
            split_con.enabled = false;
            break split_con;
        } else null;

        // Make new connections
        if (split_gene) |split_con| {
            const new_node = self.nodeCount();
            const in = Gene{
                .input = split_con.input,
                .output = new_node,
                .weight = split_con.weight,
                .enabled = true,
            };
            const out = Gene{
                .input = new_node,
                .output = split_con.output,
                .weight = 1.0,
                .enabled = true,
            };
            try self.genes.put(allocator, try gene_record.putGet(in), in);
            try self.genes.put(allocator, try gene_record.putGet(out), out);
        }
    }
}

pub fn distance(self: Self, other: Self, options: DistanceOptions) f32 {
    var weight_diff: f32 = 0.0;
    var matching: u32 = 0;
    for (self.genes.keys()) |id| {
        if (other.genes.contains(id)) {
            const a_gene = self.genes.get(id) orelse unreachable;
            const b_gene = other.genes.get(id) orelse unreachable;

            const a_weight = if (a_gene.enabled) a_gene.weight else 0.0;
            const b_weight = if (b_gene.enabled) b_gene.weight else 0.0;
            weight_diff += @abs(a_weight - b_weight);
            matching += 1;
        }
    }

    const non_matching = self.genes.count() + other.genes.count() - 2 * matching;
    const norm = @max(self.genes.count(), other.genes.count()) -| options.small_genome_bias;

    const matching_f: f32 = @floatFromInt(@max(1, matching));
    const non_matching_f: f32 = @floatFromInt(non_matching);
    const norm_f: f32 = @floatFromInt(@max(1, norm));
    return options.wight_diff_weight * weight_diff / matching_f +
        options.non_matching_weight * non_matching_f / norm_f;
}

/// The number of nodes in the network expressed by the genome.
pub fn nodeCount(self: Self) u32 {
    var node_count: u32 = 0;
    // By not checking the connection input nodes, the only nodes that never be
    // counted are the input nodes, but their ids are always smaller  than the
    // output or hidden nodes'. Node ids are enough to get the node count as
    // nodes are never removed.
    for (self.genes.values()) |gene| {
        node_count = @max(node_count, gene.output);
    }
    // Add 1 to account for 0-indexing
    return node_count + 1;
}

/// The size of the network expressed by the genome, calculated as
/// `#nodes + #enabled genes`.
pub fn size(self: Self) u32 {
    var enabled: u32 = 0;
    for (self.genes.values()) |gene| {
        if (gene.enabled) {
            enabled += 1;
        }
    }
    return self.nodeCount() + enabled;
}

/// Creates a deep copy of the genome.
pub fn clone(self: Self, allocator: Allocator) !Self {
    return .{
        .genes = try self.genes.clone(allocator),
    };
}
