using GPUCompiler, LLVM

module TestRuntime
    # dummy methods
    signal_exception() = return
    malloc(sz) = C_NULL
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

struct TestCompilerParams <: AbstractCompilerParams end
GPUCompiler.runtime_module(::CompilerJob{<:Any,TestCompilerParams}) = TestRuntime

for intr in [
        "dispatch_quadgroups_per_threadgroup", "dispatch_simdgroups_per_threadgroup",
        "quadgroup_index_in_threadgroup", "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup", "simdgroups_per_threadgroup",
        "thread_index_in_quadgroup", "thread_index_in_simdgroup", "thread_index_in_threadgroup",
        "thread_execution_width", "threads_per_simdgroup"]
    # XXX: these are also available as UInt16 (ushort)
    @eval $(Symbol(intr))() = ccall($"extern julia.air.$intr.i32", llvmcall, UInt32, ())
end

# ushort vec or uint vec
for intr in [
        "dispatch_threads_per_threadgroup",
        "grid_origin", "grid_size",
        "thread_position_in_grid", "thread_position_in_threadgroup",
        "threadgroup_position_in_grid", "threadgroups_per_grid",
        "threads_per_grid", "threads_per_threadgroup"]
    # XXX: these are also available as UInt16 (ushort)
    @eval $(Symbol(intr * "_1d"))() = ccall($"extern julia.air.$intr.i32", llvmcall, UInt32, ())
    @eval $(Symbol(intr * "_2d"))() = ccall($"extern julia.air.$intr.v2i32", llvmcall, NTuple{2, VecElement{UInt32}}, ())
    @eval $(Symbol(intr * "_3d"))() = ccall($"extern julia.air.$intr.v3i32", llvmcall, NTuple{3, VecElement{UInt32}}, ())
end

function kernel()
    child()
    return
end

@noinline child() = (thread_position_in_grid_2d(); return)

function main()
    source = FunctionSpec(kernel)
    target = MetalCompilerTarget()
    params = TestCompilerParams()
    job = CompilerJob(target, source, params)

    println(GPUCompiler.compile(:asm, job)[1])
end

isinteractive() || main()
