const std = @import("std");

const zmai = @import("zmai");
const Genome = neat.Genome;
const neat = zmai.genetic.neat;
const NN = neat.NN;
const Trainer = neat.Trainer;

// XOR truth table
const x = [_][]const f32{
    &.{ 0, 0 },
    &.{ 0, 1 },
    &.{ 1, 0 },
    &.{ 1, 1 },
};
const y = [_][]const f32{
    &.{0},
    &.{1},
    &.{1},
    &.{0},
};

// Trains a NEAT population to learn the binary XOR function.
pub fn main() !void {
    const POP_SIZE = 50;
    const GENERATIONS = 50;

    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    const allocator = debug_allocator.allocator();
    defer _ = debug_allocator.deinit();

    // Define the population
    zmai.setRandomSeed(23);
    const nn_options: NN.InitOptions = .{
        .input_count = 2,
        .output_count = 1,
        .hidden_activation = .relu,
        .output_activation = .tanh,
    };
    var trainer: Trainer = try .init(
        allocator,
        POP_SIZE,
        1.0,
        .{
            .species_target = 4,
            .nn_options = nn_options,
            .mutate_options = .{
                .node_add_prob = 0.06,
            },
        },
    );
    defer trainer.deinit();

    // Train the population
    const fitnesses = try allocator.alloc(f64, POP_SIZE);
    defer allocator.free(fitnesses);
    for (0..GENERATIONS) |i| {
        const start = std.time.nanoTimestamp();

        var total_fitness: f64 = 0;
        var max_fitness: f64 = 0;
        for (trainer.population, 0..) |genome, j| {
            const nn: NN = try .init(
                allocator,
                genome,
                nn_options,
                &.{},
            );
            defer nn.deinit(allocator);

            fitnesses[j] = evalFitness(nn);
            total_fitness += fitnesses[j];
            max_fitness = @max(max_fitness, fitnesses[j]);
        }

        if (i != GENERATIONS - 1) {
            try trainer.nextGeneration(fitnesses);
        }

        const time_taken: u64 = @intCast(std.time.nanoTimestamp() - start);
        std.debug.print("Epoch: {d:>2}, Species: {}, Avg fit: {d:.3}, Max fit: {d:.3}, Time: {}\n", .{
            i,
            trainer.species.len,
            total_fitness / POP_SIZE,
            max_fitness,
            std.fmt.fmtDuration(time_taken),
        });
    }

    // Save the trained population
    {
        const obj: Trainer.TrainerJson = try .init(allocator, trainer);
        defer obj.deinit(allocator);

        try std.fs.cwd().makePath(".examples-data/");
        const file = try std.fs.cwd().createFile(".examples-data/xor.json", .{});
        defer file.close();

        try std.json.stringify(obj, .{}, file.writer());
    }

    // Print the fittest NN's predictions
    const index = std.mem.indexOfMax(f64, fitnesses);
    const nn: NN = try .init(
        allocator,
        trainer.population[index],
        nn_options,
        &.{},
    );
    defer nn.deinit(allocator);

    std.debug.print("\nFittest size: {}\n", .{trainer.population[index].size()});
    for (x) |in| {
        var out: [1]f32 = undefined;
        nn.predict(in, &out);
        std.debug.print("{d} ^ {d} -> {d}\n", .{ in[0], in[1], out[0] });
    }
}

fn evalFitness(nn: NN) f64 {
    var fitness: f64 = 0;
    for (x, y) |in, actual| {
        var out: [1]f32 = undefined;
        nn.predict(in, &out);

        // Mean absolute error
        const err = @abs(actual[0] - out[0]);
        fitness += @max(0, 1 - err);
    }
    return fitness / x.len;
}
