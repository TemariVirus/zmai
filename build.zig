const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "zmai",
        .root_source_file = .{ .path = "src/examples/minst.zig" },
        .target = target,
        .optimize = optimize,
    });

    const zmai_module = b.addModule("zmai", .{
        .root_source_file = .{ .path = "src/root.zig" },
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
}
