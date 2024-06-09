const std = @import("std");
const Build = std.Build;

pub fn build(b: *Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zmai",
        .root_source_file = lazyPath(b, "src/examples/minst.zig"),
        .target = target,
        .optimize = optimize,
    });

    const zmai_module = b.addModule("zmai", .{
        .root_source_file = lazyPath(b, "src/root.zig"),
        .imports = &.{},
    });

    exe.root_module.addImport("zmai", zmai_module);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    buildTests(b);
}

fn buildTests(b: *Build) void {
    const lib_tests = b.addTest(.{
        .root_source_file = lazyPath(b, "src/root.zig"),
    });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_lib_tests.step);
}

fn lazyPath(b: *Build, path: []const u8) Build.LazyPath {
    return .{
        .src_path = .{
            .owner = b,
            .sub_path = path,
        },
    };
}
