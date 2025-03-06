const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zmai_module = b.addModule("zmai", .{
        .root_source_file = b.path("src/root.zig"),
    });

    if (b.option([]const u8, "example", "The example to build")) |name| {
        const exe = try buildExample(b, name, target, optimize, zmai_module);

        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(b.getInstallStep());
        if (b.args) |args| {
            run_cmd.addArgs(args);
        }

        const run_step = b.step("run", "Run the app");
        run_step.dependOn(&run_cmd.step);
    }

    buildTests(b);
}

fn buildExample(
    b: *Build,
    name: []const u8,
    target: Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    zmai_module: *Build.Module,
) !*Build.Step.Compile {
    var buf: [255]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "src/examples/{s}.zig", .{name});

    const exe = b.addExecutable(.{
        .name = "zmai",
        .root_source_file = b.path(path),
        .target = target,
        .optimize = optimize,
    });
    exe.root_module.addImport("zmai", zmai_module);

    return exe;
}

fn buildTests(b: *Build) void {
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);
}
