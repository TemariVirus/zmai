const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const math = std.math;

const random = @import("../../root.zig").random;

const neat = @import("../neat.zig");
const ActivationType = neat.ActivationType;
const Gene = Genome.Gene;
const Genome = neat.Genome;
const GenomeJson = Genome.GenomeJson;
const NN = neat.NN;

const Self = @This();

pub const GeneRecord = struct {
    pub const GeneMap = std.AutoHashMap(struct { in: u32, out: u32 }, u32);

    innovation_counter: u32 = 0,
    genes: GeneMap,

    pub fn init(allocator: Allocator) GeneRecord {
        return .{
            .genes = GeneMap.init(allocator),
        };
    }

    pub fn deinit(self: *GeneRecord) void {
        self.genes.deinit();
    }

    pub fn get(self: GeneRecord, gene: Gene) ?u32 {
        return self.genes.get(.{
            .in = gene.input,
            .out = gene.output,
        });
    }

    pub fn putGet(self: *GeneRecord, gene: Gene) !u32 {
        const result = try self.genes.getOrPut(.{
            .in = gene.input,
            .out = gene.output,
        });
        if (!result.found_existing) {
            result.value_ptr.* = self.innovation_counter;
            self.innovation_counter += 1;
        }
        return result.value_ptr.*;
    }
};

/// A JSON serializable representation of a trainer and its population.
pub const TrainerJson = struct {
    generation: u64,
    compat_tresh: f32,
    options: Options,
    species: []const GenomeJson,
    population: []const GenomeJson,

    /// Creates a JSON serializable object from a trainer without copying
    /// genome data. Modifying the trainer may invalidate the object.
    pub fn init(allocator: Allocator, trainer: Self) !TrainerJson {
        const species = try allocator.alloc(GenomeJson, trainer.species.len);
        errdefer allocator.free(species);
        for (0..trainer.species.len) |i| {
            species[i] = GenomeJson.initNoCopy(trainer.species[i]);
        }

        const population = try allocator.alloc(GenomeJson, trainer.population.len);
        errdefer allocator.free(population);
        for (0..trainer.population.len) |i| {
            population[i] = GenomeJson.initNoCopy(trainer.population[i]);
        }

        return .{
            .generation = trainer.generation,
            .compat_tresh = trainer.compat_tresh,
            .options = trainer.options,
            .species = species,
            .population = population,
        };
    }

    pub fn deinit(self: TrainerJson, allocator: Allocator) void {
        allocator.free(self.species);
        allocator.free(self.population);
    }
};

// TODO: cross-species mating
// TODO: split up
pub const Options = struct {
    species_target: usize = 10,
    species_try_times: u32 = 5,
    /// This value should be at least 1.
    compat_mod: f32 = 1.5,
    elite_percent: f32 = 0.3,
    retain_top: usize = 1,

    nn_options: NN.InitOptions,
    distance_options: Genome.DistanceOptions = .{},
    mate_options: Genome.MateOptions = .{},
    mutate_options: Genome.MutateOptions = .{},
};

allocator: Allocator,
generation: u64 = 0,
compat_tresh: f32,
gene_record: GeneRecord,
options: Options,
species: []Genome = &.{},
population: []Genome,

pub fn init(allocator: Allocator, population_size: usize, compat_tresh: f32, options: Options) !Self {
    var gene_record = GeneRecord.init(allocator);
    errdefer gene_record.deinit();

    var population = try std.ArrayListUnmanaged(Genome)
        .initCapacity(allocator, population_size);
    errdefer {
        for (0..population.items.len) |i| {
            population.items[i].deinit(allocator);
        }
        population.deinit(allocator);
    }

    for (0..population_size) |_| {
        population.appendAssumeCapacity(try Genome.init(
            allocator,
            options.nn_options.input_count,
            options.nn_options.output_count,
            &gene_record,
        ));
    }

    return .{
        .allocator = allocator,
        .compat_tresh = compat_tresh,
        .gene_record = gene_record,
        .options = options,
        .population = population.items,
    };
}

/// Deserializes a trainer from a JSON object.
pub fn from(allocator: Allocator, obj: TrainerJson) !Self {
    var gene_record = GeneRecord.init(allocator);
    errdefer gene_record.deinit();

    var species = try std.ArrayListUnmanaged(Genome)
        .initCapacity(allocator, obj.species.len);
    errdefer {
        for (0..species.items.len) |i| {
            species.items[i].deinit(allocator);
        }
        species.deinit(allocator);
    }
    for (obj.species) |genome| {
        species.appendAssumeCapacity(try Genome.from(allocator, genome, &gene_record));
    }

    var population = try std.ArrayListUnmanaged(Genome)
        .initCapacity(allocator, obj.population.len);
    errdefer {
        for (0..population.items.len) |i| {
            population.items[i].deinit(allocator);
        }
        population.deinit(allocator);
    }
    for (obj.population) |genome| {
        population.appendAssumeCapacity(try Genome.from(allocator, genome, &gene_record));
    }

    return .{
        .allocator = allocator,
        .generation = obj.generation,
        .compat_tresh = obj.compat_tresh,
        .gene_record = gene_record,
        .options = obj.options,
        .species = species.items,
        .population = population.items,
    };
}

pub fn deinit(self: *Self) void {
    self.gene_record.deinit();

    for (0..self.species.len) |i| {
        self.species[i].deinit(self.allocator);
    }
    self.allocator.free(self.species);

    for (0..self.population.len) |i| {
        self.population[i].deinit(self.allocator);
    }
    self.allocator.free(self.population);
}

pub fn nextGeneration(self: *Self, fitnesses: []const f64) !void {
    assert(fitnesses.len == self.population.len);
    // Fitness must be a non-negative number
    for (fitnesses) |f| {
        if (!math.isFinite(f)) {
            return error.NonFiniteFitness;
        }
        if (f < 0) {
            return error.NegativeFitness;
        }
    }

    const old_compat_tresh = self.compat_tresh;
    errdefer self.compat_tresh = old_compat_tresh;

    const species, const speciated = for (0..self.options.species_try_times) |i| {
        const species, const speciated = try speciate(
            self.allocator,
            self.species,
            self.population,
            self.compat_tresh,
            self.options.distance_options,
        );
        if (species.len == self.options.species_target) {
            break .{ species, speciated };
        }

        // Adjust compat_tresh
        const compat_power = 1 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(self.options.species_try_times));
        const compat_mod = math.pow(f32, self.options.compat_mod, compat_power);
        if (species.len > self.options.species_target) {
            self.compat_tresh *= compat_mod;
        } else {
            self.compat_tresh /= compat_mod;
        }

        for (0..species.len) |j| {
            species[j].deinit(self.allocator);
        }
        self.allocator.free(species);
        for (0..speciated.len) |j| {
            speciated[j].deinit(self.allocator);
        }
        self.allocator.free(speciated);
    } else try speciate(
        self.allocator,
        self.species,
        self.population,
        self.compat_tresh,
        self.options.distance_options,
    );
    assert(species.len == speciated.len);
    defer {
        for (0..speciated.len) |i| {
            speciated[i].deinit(self.allocator);
        }
        self.allocator.free(speciated);
    }

    // Mating
    const offspring_counts = try offspringCounts(self.allocator, speciated, fitnesses);
    defer self.allocator.free(offspring_counts);
    var population = std.ArrayList(Genome).init(self.allocator);
    try population.ensureTotalCapacityPrecise(self.population.len);
    errdefer {
        for (0..population.items.len) |i| {
            population.items[i].deinit(self.allocator);
        }
        population.deinit();
    }

    for (speciated, offspring_counts) |s, count| {
        const lessThanFn = struct {
            fn lessThanFn(fits: []const f64, lhs: usize, rhs: usize) bool {
                // Use greater than to sort in descending order
                return fits[lhs] > fits[rhs];
            }
        }.lessThanFn;
        std.sort.pdq(usize, s.items, fitnesses, lessThanFn);

        // Add retain_top without mutating
        for (0..@min(s.items.len, @min(count, self.options.retain_top))) |i| {
            population.appendAssumeCapacity(
                try self.population[s.items[i]].clone(self.allocator),
            );
        }

        // Only top few can reproduce
        const elite_count = @as(f32, @floatFromInt(s.items.len)) * self.options.elite_percent;
        const elites = s.items[0..@intFromFloat(@ceil(elite_count))];

        // Prepare roulette wheel (fitter parents have higher chance of mating)
        const wheel = try self.allocator.alloc(f64, elites.len);
        defer self.allocator.free(wheel);
        for (elites, 0..) |genome_i, i| {
            wheel[i] = fitnesses[genome_i];
        }

        // Make children
        for (0..count -| @min(s.items.len, self.options.retain_top)) |_| {
            const index1 = elites[random.weightedIndex(f64, wheel)];
            const parent1 = self.population[index1];

            const index2 = elites[random.weightedIndex(f64, wheel)];
            const parent2 = self.population[index2];

            var child = try parent1.mateWith(self.allocator, parent2, self.options.mate_options);
            errdefer child.deinit(self.allocator);
            try child.mutate(self.allocator, &self.gene_record, self.options.mutate_options);
            population.appendAssumeCapacity(child);
        }
    }

    self.generation += 1;
    for (0..self.species.len) |i| {
        self.species[i].deinit(self.allocator);
    }
    self.allocator.free(self.species);
    self.species = species;

    const old_population = self.population;
    self.population = try population.toOwnedSlice();
    for (0..old_population.len) |i| {
        old_population[i].deinit(self.allocator);
    }
    self.allocator.free(old_population);
}

pub const SpeciesList = std.ArrayListUnmanaged(usize);
pub fn speciate(
    allocator: Allocator,
    old_species: []const Genome,
    population: []const Genome,
    compat_tresh: f32,
    options: Genome.DistanceOptions,
) !struct { []Genome, []SpeciesList } {
    var species = std.ArrayListUnmanaged(Genome){};
    try species.ensureTotalCapacity(allocator, old_species.len);
    errdefer {
        for (0..species.items.len) |i| {
            species.items[i].deinit(allocator);
        }
        species.deinit(allocator);
    }
    for (old_species) |genome| {
        species.appendAssumeCapacity(try genome.clone(allocator));
    }

    var speciated = std.ArrayListUnmanaged(SpeciesList){};
    try speciated.appendNTimes(allocator, SpeciesList{}, species.items.len);
    errdefer {
        for (0..speciated.items.len) |i| {
            speciated.items[i].deinit(allocator);
        }
        speciated.deinit(allocator);
    }

    for (population, 0..) |genome, i| {
        for (species.items, 0..) |repr, species_i| {
            if (repr.distance(genome, options) < compat_tresh) {
                try speciated.items[species_i].append(allocator, i);
                break;
            }
        } else {
            // Make new species if no match
            try species.append(allocator, try genome.clone(allocator));
            try speciated.append(allocator, SpeciesList{});
            try speciated.items[speciated.items.len - 1].append(allocator, i);
        }
    }

    // Remove empty species
    var i: usize = 0;
    while (i < species.items.len) {
        if (speciated.items[i].items.len == 0) {
            var s = speciated.swapRemove(i);
            s.deinit(allocator);
            var genome = species.swapRemove(i);
            genome.deinit(allocator);
        } else {
            i += 1;
        }
    }

    return .{
        try species.toOwnedSlice(allocator),
        try speciated.toOwnedSlice(allocator),
    };
}

/// Explicit fitness sharing.
pub fn offspringCounts(
    allocator: Allocator,
    speciated: []const SpeciesList,
    fitnesses: []const f64,
) ![]const u64 {
    // Find mean fitnesses
    const mean_fitnesses = try allocator.alloc(f64, speciated.len);
    var total_mean_fitness: f64 = 0;
    for (0..speciated.len) |i| {
        mean_fitnesses[i] = 0.0;
        for (speciated[i].items) |genome_i| {
            mean_fitnesses[i] += fitnesses[genome_i];
        }
        mean_fitnesses[i] /= @floatFromInt(speciated[i].items.len);
        total_mean_fitness += mean_fitnesses[i];
    }

    // Calculate amount of babies
    const offspring_counts: []u64 = @ptrCast(mean_fitnesses);
    var total_offspring: usize = 0;
    for (0..offspring_counts.len) |i| {
        const count = @floor(mean_fitnesses[i] * @as(f64, @floatFromInt(fitnesses.len)) / total_mean_fitness);
        offspring_counts[i] = @intFromFloat(count);
        total_offspring += offspring_counts[i];
    }

    // Distribute extra babies evenly
    for (0..offspring_counts.len) |i| {
        if (total_offspring == fitnesses.len) {
            break;
        }
        offspring_counts[i] += 1;
        total_offspring += 1;
    }
    assert(total_offspring == fitnesses.len);

    return offspring_counts;
}
